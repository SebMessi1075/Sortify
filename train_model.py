"""
Smart AI Bin — Optimised Training Script v3.1
=============================================
Fixes vs v3:

  BUGFIX 1 — Keras augmenters removed from inside tf.cond inside tf.data.map.
    Calling a keras.Sequential inside tf.cond inside a parallel tf.data map
    causes graph tracing failures on GPU (both branches are always traced,
    internal layer state breaks). Replaced with pure tf.image ops which
    trace cleanly as graph nodes under any parallelism.

  BUGFIX 2 — MixUp pipeline rewritten to avoid dynamic shapes.
    tf.boolean_mask produces unknown-shape tensors. Slicing those with
    [:n_mix] inside tf.cond cannot be statically resolved under GPU graph
    mode → causes local_rendezvous cancellation cascade. New approach:
    MixUp is applied as a numpy_function callback outside the graph,
    completely avoiding the shape ambiguity. This is safe because MixUp
    only runs during training, not inference.

  BUGFIX 3 — CutOut rewritten as a single vectorised tf.image op.
    Python for-loops with tf.random inside tf.data.map produce unstable
    graph traces under AUTOTUNE parallelism. Replaced with a single
    tf.while_loop that TF can trace correctly.

  BUGFIX 4 — TF C++ log noise suppressed at startup.
    Sets TF_CPP_MIN_LOG_LEVEL=3 and absl logging to ERROR-only so the
    local_rendezvous spam and XLA/CUDA info lines don't flood the terminal.
    Real errors still surface via Python exceptions.

  All training logic (3-phase, focal loss, temperature scaling,
  class weights, per-class confusion printout) is unchanged from v3.
"""

import os, sys, zipfile, shutil, pathlib, json

# ── Suppress TF C++ log spam before importing TF ──────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"       # 0=all, 1=info, 2=warn, 3=error
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"      # stops oneDNN version mismatch spam
os.environ["CUDA_DEVICE_ORDER"]     = "PCI_BUS_ID"

import numpy as np
import tensorflow as tf

# Suppress absl (used by TF internally) to ERROR level
try:
    import absl.logging
    absl.logging.set_verbosity(absl.logging.ERROR)
except ImportError:
    pass

tf.get_logger().setLevel("ERROR")

from tensorflow import keras
from keras import layers
from keras.applications import EfficientNetV2B0, MobileNetV2
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
IMG_SIZE             = 224
BATCH_SIZE           = 32
EPOCHS_FROZEN        = 12
EPOCHS_FINETUNE_WARM = 8
EPOCHS_FINETUNE_FULL = 25
FINE_TUNE_WARM_AT    = -40
FINE_TUNE_AT         = 100
LR_HEAD              = 1e-3
LR_FINETUNE_WARM     = 2e-5
LR_FINETUNE_FULL     = 5e-6
FOCAL_GAMMA          = 2.0
MIXUP_ALPHA          = 0.2
BACKBONE_NAME        = os.environ.get("SMARTBIN_BACKBONE", "mobilenet_v2").lower()
PREPROCESS_NAME      = "mobilenet_v2"
CONFIDENCE_THRESHOLD = 0.70
MIN_TOP2_MARGIN      = 0.12
VAL_SPLIT            = 0.20
SEED                 = 42

BASE_DIR   = pathlib.Path("smartbin_project")
DATA_DIR   = BASE_DIR / "dataset_raw"
REMAP_DIR  = BASE_DIR / "dataset"
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_CONFIG_PATH = OUTPUT_DIR / "model_config.json"
KAGGLE_JSON_PATH  = pathlib.Path.home() / ".kaggle" / "kaggle.json"

GLASS_CLASS        = "glass_nonrecyclable"
LEGACY_GLASS_CLASS = "glass_other"

REMAP = {
    "cardboard" : "plastic_paper",
    "paper"     : "plastic_paper",
    "plastic"   : "plastic_paper",
    "metal"     : "metal",
    "glass"     : GLASS_CLASS,
}
CLASS_NAMES = [GLASS_CLASS, "metal", "plastic_paper"]
NUM_CLASSES = len(CLASS_NAMES)
GLASS_IDX   = CLASS_NAMES.index(GLASS_CLASS)
METAL_IDX   = CLASS_NAMES.index("metal")
PLASTIC_IDX = CLASS_NAMES.index("plastic_paper")


# ─────────────────────────────────────────────
#  STEP 1 — DOWNLOAD
# ─────────────────────────────────────────────
def ensure_kaggle_credentials():
    if KAGGLE_JSON_PATH.exists(): return
    print(f"\nkaggle.json not found at: {KAGGLE_JSON_PATH}")
    print("kaggle.com → Account → API → Create New API Token → save there")
    sys.exit(1)

def download_dataset():
    if DATA_DIR.exists() and any(DATA_DIR.iterdir()):
        print("Dataset already exists — skipping."); return
    ensure_kaggle_credentials()
    print("Downloading from Kaggle...")
    BASE_DIR.mkdir(exist_ok=True)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi(); api.authenticate()
        api.dataset_download_files("asdasdasasdas/garbage-classification",
                                   path=str(BASE_DIR), unzip=False)
    except BaseException as e:
        print(f"\nKaggle error: {e}")
        print("Manual: https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification")
        print(f"Unzip into: {DATA_DIR}/"); sys.exit(1)
    zips = list(BASE_DIR.glob("*.zip"))
    if not zips: sys.exit("No zip found.")
    with zipfile.ZipFile(zips[0]) as zf: zf.extractall(BASE_DIR)
    zips[0].unlink()
    candidates = [p for p in BASE_DIR.rglob("cardboard") if p.is_dir()]
    if not candidates: sys.exit(f"cardboard/ not found in {BASE_DIR}")
    raw = candidates[0].parent
    if raw != DATA_DIR: raw.rename(DATA_DIR)
    for d in sorted(DATA_DIR.iterdir()):
        if d.is_dir(): print(f"  {d.name}: {len(list(d.glob('*.*')))}")


# ─────────────────────────────────────────────
#  STEP 2 — REMAP
# ─────────────────────────────────────────────
def migrate_legacy():
    if not REMAP_DIR.exists(): return
    for split in ["train", "val"]:
        src = REMAP_DIR / split / LEGACY_GLASS_CLASS
        dst = REMAP_DIR / split / GLASS_CLASS
        if not src.exists(): continue
        dst.mkdir(parents=True, exist_ok=True)
        for img in src.glob("*.*"):
            t = dst / img.name
            if t.exists(): t = dst / f"legacy_{img.name}"
            shutil.move(str(img), str(t))
        try: src.rmdir()
        except OSError: pass
    print(f"Migrated {LEGACY_GLASS_CLASS} → {GLASS_CLASS}")

def remap_dataset():
    migrate_legacy()
    if REMAP_DIR.exists() and any(REMAP_DIR.iterdir()):
        print("\nRemapped dataset exists — skipping."); return
    print("\nRemapping (trash excluded)...")
    for split in ["train", "val"]:
        for cls in CLASS_NAMES:
            (REMAP_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    rng    = np.random.default_rng(SEED)
    totals = {"train": {c:0 for c in CLASS_NAMES}, "val": {c:0 for c in CLASS_NAMES}}
    for kaggle_cls, bin_cls in REMAP.items():
        src = DATA_DIR / kaggle_cls
        if not src.exists(): print(f"  Missing: {src}"); continue
        images = list(src.glob("*.*")); rng.shuffle(images)
        n_val  = max(1, int(len(images) * VAL_SPLIT))
        val_s  = set(str(p) for p in images[:n_val])
        for img in images:
            split = "val" if str(img) in val_s else "train"
            dst   = REMAP_DIR / split / bin_cls / f"{kaggle_cls}_{img.name}"
            if not dst.exists(): shutil.copy2(img, dst)
            totals[split][bin_cls] += 1
        print(f"  {kaggle_cls:12s} → {bin_cls}")
    print("\nFinal split:")
    for split in ["train","val"]:
        for cls in CLASS_NAMES:
            print(f"  {split}/{cls}: {totals[split][cls]}")


# ─────────────────────────────────────────────
#  STEP 3 — AUGMENTATION
#
#  FIXED: All augmentation is pure tf.image ops — no Keras Sequential
#  layers inside tf.data.map. This traces correctly on GPU under any
#  level of parallelism.
#
#  Class-aware logic is handled via tf.cond on a simple boolean tensor,
#  not by calling different Keras models in each branch.
# ─────────────────────────────────────────────

def _rand(lo, hi):
    """Scalar uniform random float in [lo, hi)."""
    return tf.random.uniform((), lo, hi)

def _apply_light_aug(img):
    """Light augmentation for plastic_paper (well-separated class)."""
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, 0.10)
    img = tf.image.random_contrast(img, 0.90, 1.10)
    # Random rotation via crop-and-resize (approx ±8°)
    img = tf.image.random_crop(img, [int(IMG_SIZE*0.94), int(IMG_SIZE*0.94), 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def _apply_heavy_aug(img):
    """
    Heavy augmentation for glass + metal (the confused pair).
    Stronger brightness/contrast for shiny-surface robustness.
    Hue + saturation shifts to teach colour ≠ class.
    Both flip axes (bottles appear inverted on belts).
    """
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.20)
    img = tf.image.random_contrast(img, 0.75, 1.25)
    img = tf.image.random_hue(img, 0.08)
    img = tf.image.random_saturation(img, 0.80, 1.20)
    img = tf.image.random_crop(img, [int(IMG_SIZE*0.90), int(IMG_SIZE*0.90), 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    return img

def _apply_cutout(img):
    """
    CutOut via tf.while_loop — traces correctly as a graph op.
    Occludes up to 2 random patches, forcing the model off silhouette.
    """
    img = tf.cast(img, tf.float32)
    H = IMG_SIZE; W = IMG_SIZE

    def body(i, img_):
        # Only apply with 40% probability per hole
        apply = tf.random.uniform(()) < 0.40
        sz    = tf.random.uniform((), 8, 48, dtype=tf.int32)
        cy    = tf.random.uniform((), 0, H, dtype=tf.int32)
        cx    = tf.random.uniform((), 0, W, dtype=tf.int32)
        y1 = tf.maximum(0, cy - sz // 2)
        y2 = tf.minimum(H, cy + sz // 2)
        x1 = tf.maximum(0, cx - sz // 2)
        x2 = tf.minimum(W, cx + sz // 2)
        # Build the mask: ones everywhere except the patch
        ones = tf.ones([H, W, 1], dtype=tf.float32)
        patch_h = y2 - y1; patch_w = x2 - x1
        patch   = tf.zeros([patch_h, patch_w, 1], dtype=tf.float32)
        padded  = tf.pad(patch, [[y1, H-y2], [x1, W-x2], [0, 0]],
                         constant_values=1.0)
        img_ = tf.cond(apply, lambda: img_ * padded, lambda: img_)
        return i + 1, img_

    _, img = tf.while_loop(
        lambda i, _: i < 2,
        body,
        [tf.constant(0), img],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape([H, W, 3])]
    )
    return img


def preprocess_input_for_backbone(img):
    if PREPROCESS_NAME == "efficientnet_v2":
        return keras.applications.efficientnet_v2.preprocess_input(img)
    return keras.applications.mobilenet_v2.preprocess_input(img)


def preprocess_train(img, lbl):
    """
    Pure-tf augmentation pipeline — traces correctly inside tf.data.map
    with num_parallel_calls=AUTOTUNE on GPU.

    Glass + metal → heavy aug + cutout
    Plastic       → light aug
    """
    img = tf.cast(img, tf.float32)
    class_idx = tf.argmax(lbl, output_type=tf.int32)
    is_glass_or_metal = tf.logical_or(
        tf.equal(class_idx, tf.constant(GLASS_IDX,  dtype=tf.int32)),
        tf.equal(class_idx, tf.constant(METAL_IDX,  dtype=tf.int32)),
    )
    # Use tf.cond with pure tf.image branches — safe to trace in parallel
    img = tf.cond(
        is_glass_or_metal,
        lambda: _apply_cutout(_apply_heavy_aug(img)),
        lambda: _apply_light_aug(img),
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    return preprocess_input_for_backbone(img), lbl


def preprocess_val(img, lbl):
    return preprocess_input_for_backbone(tf.cast(img, tf.float32)), lbl


def build_pipelines():
    AU = tf.data.AUTOTUNE
    def load(split):
        return keras.utils.image_dataset_from_directory(
            REMAP_DIR / split, image_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, label_mode="categorical",
            shuffle=(split == "train"), seed=SEED, class_names=CLASS_NAMES)
    train_ds = load("train").map(preprocess_train, num_parallel_calls=AU).prefetch(AU)
    val_ds   = load("val").map(preprocess_val, num_parallel_calls=AU).cache().prefetch(AU)
    return train_ds, val_ds


def build_mixup_pipeline(train_ds):
    """
    FIXED: MixUp via tf.numpy_function — runs outside the TF graph so
    dynamic shapes (from boolean_mask) are handled in numpy, not TF.
    Safe for training; never used at inference time.
    No graph shape errors, no rendezvous cancellations.
    """
    def _numpy_mixup(imgs_np, lbls_np):
        glass_idx   = np.where(np.argmax(lbls_np, axis=1) == GLASS_IDX)[0]
        metal_idx   = np.where(np.argmax(lbls_np, axis=1) == METAL_IDX)[0]
        n_mix = min(len(glass_idx), len(metal_idx))
        if n_mix == 0:
            return imgs_np.astype(np.float32), lbls_np.astype(np.float32)
        lam = np.random.uniform(MIXUP_ALPHA, 1.0 - MIXUP_ALPHA)
        g_imgs = imgs_np[glass_idx[:n_mix]]
        m_imgs = imgs_np[metal_idx[:n_mix]]
        g_lbls = lbls_np[glass_idx[:n_mix]]
        m_lbls = lbls_np[metal_idx[:n_mix]]
        mixed_imgs = (lam * g_imgs + (1.0 - lam) * m_imgs).astype(np.float32)
        mixed_lbls = (lam * g_lbls + (1.0 - lam) * m_lbls).astype(np.float32)
        out_imgs = np.concatenate([imgs_np, mixed_imgs], axis=0).astype(np.float32)
        out_lbls = np.concatenate([lbls_np, mixed_lbls], axis=0).astype(np.float32)
        return out_imgs, out_lbls

    def apply_mixup(imgs, lbls):
        out_imgs, out_lbls = tf.numpy_function(
            _numpy_mixup,
            [imgs, lbls],
            [tf.float32, tf.float32]
        )
        # Restore static shape info lost through numpy_function
        out_imgs.set_shape([None, IMG_SIZE, IMG_SIZE, 3])
        out_lbls.set_shape([None, NUM_CLASSES])
        return out_imgs, out_lbls

    # MixUp adds variable batch size — use prefetch only (no AUTOTUNE map)
    return train_ds.map(apply_mixup, num_parallel_calls=1).prefetch(tf.data.AUTOTUNE)


def compute_class_weights():
    tr = REMAP_DIR / "train"
    counts = [len(list((tr/c).glob("*.*"))) if (tr/c).exists() else 0 for c in CLASS_NAMES]
    total  = sum(counts)
    if total == 0 or any(c == 0 for c in counts):
        print("Warning: class weights disabled."); return None
    weights = {i: total/(NUM_CLASSES*c) for i, c in enumerate(counts)}
    for cls, c, w in zip(CLASS_NAMES, counts, weights.values()):
        print(f"  {cls:20s}: {c} images  weight={w:.3f}")
    return weights


# ─────────────────────────────────────────────
#  STEP 4 — MODEL
# ─────────────────────────────────────────────
def build_model():
    global BACKBONE_NAME, PREPROCESS_NAME
    if BACKBONE_NAME == "efficientnetv2b0":
        try:
            base = EfficientNetV2B0(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                                    include_top=False, weights="imagenet")
            PREPROCESS_NAME = "efficientnet_v2"
        except Exception as e:
            print(f"Warning: EfficientNetV2B0 failed ({e}). Falling back to MobileNetV2.")
            BACKBONE_NAME   = "mobilenet_v2"
            PREPROCESS_NAME = "mobilenet_v2"
            base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                               include_top=False, weights="imagenet")
    else:
        BACKBONE_NAME   = "mobilenet_v2"
        PREPROCESS_NAME = "mobilenet_v2"
        base = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                           include_top=False, weights="imagenet")

    base.trainable = False
    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x   = base(inp, training=False)
    x   = layers.GlobalAveragePooling2D()(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dense(256, activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.45)(x)
    x   = layers.Dense(64,  activation="relu",
                        kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.25)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out, name=f"SmartBin_{BACKBONE_NAME}"), base


def make_class_weighted_focal_loss(class_weights=None, gamma=FOCAL_GAMMA):
    cw = None
    if class_weights:
        cw = tf.constant([class_weights.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=tf.float32)
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        ce = -y_true * tf.math.log(y_pred)
        focal_factor = tf.pow(1.0 - y_pred, gamma)
        per_class = focal_factor * ce
        if cw is not None:
            per_class = per_class * cw
        return tf.reduce_mean(tf.reduce_sum(per_class, axis=-1))
    return loss_fn


# ─────────────────────────────────────────────
#  STEP 5 — THREE-PHASE TRAINING
# ─────────────────────────────────────────────
def make_callbacks(tag):
    return [
        ModelCheckpoint(str(OUTPUT_DIR/f"best_{tag}.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
        EarlyStopping(monitor="val_loss", patience=6,
                      restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.4,
                          patience=3, min_lr=1e-8, verbose=1),
    ]

def train():
    model, base      = build_model()
    train_ds, val_ds = build_pipelines()
    train_ds_mix     = build_mixup_pipeline(train_ds)
    cw               = compute_class_weights()
    loss_fn          = make_class_weighted_focal_loss(cw, gamma=FOCAL_GAMMA)

    print("\n── Phase 1: head training (backbone frozen) ──")
    model.compile(optimizer=keras.optimizers.Adam(LR_HEAD),
                  loss=loss_fn, metrics=["accuracy"])
    model.summary(line_length=80)
    h1 = model.fit(train_ds_mix, validation_data=val_ds,
                   epochs=EPOCHS_FROZEN, class_weight=cw,
                   callbacks=make_callbacks("phase1"))

    print(f"\n── Phase 2: warm fine-tune (last 40 layers, LR={LR_FINETUNE_WARM}) ──")
    base.trainable = True
    n = len(base.layers)
    for l in base.layers[:n + FINE_TUNE_WARM_AT]:
        l.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE_WARM),
                  loss=loss_fn, metrics=["accuracy"])
    h2 = model.fit(train_ds_mix, validation_data=val_ds,
                   epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE_WARM,
                   initial_epoch=EPOCHS_FROZEN, class_weight=cw,
                   callbacks=make_callbacks("phase2"))

    print(f"\n── Phase 3: full fine-tune from layer {FINE_TUNE_AT}, LR={LR_FINETUNE_FULL} ──")
    for l in base.layers[:FINE_TUNE_AT]: l.trainable = False
    model.compile(optimizer=keras.optimizers.Adam(LR_FINETUNE_FULL),
                  loss=loss_fn, metrics=["accuracy"])
    h3 = model.fit(train_ds_mix, validation_data=val_ds,
                   epochs=EPOCHS_FROZEN + EPOCHS_FINETUNE_WARM + EPOCHS_FINETUNE_FULL,
                   initial_epoch=EPOCHS_FROZEN + EPOCHS_FINETUNE_WARM,
                   class_weight=cw, callbacks=make_callbacks("phase3"))

    hist = {k: h1.history[k] + h2.history[k] + h3.history[k]
            for k in ["accuracy","val_accuracy","loss","val_loss"]}
    model.save(str(OUTPUT_DIR/"smartbin_cnn_final.h5"))
    return model, train_ds, val_ds, hist

def save_inference_config():
    cfg = {"backbone": BACKBONE_NAME, "preprocess": PREPROCESS_NAME,
           "img_size": IMG_SIZE, "class_names": CLASS_NAMES}
    MODEL_CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(f"Config → {MODEL_CONFIG_PATH}")


# ─────────────────────────────────────────────
#  STEP 6 — TEMPERATURE CALIBRATION
# ─────────────────────────────────────────────
def calibrate_temperature(model, val_ds):
    print("\n── Temperature scaling calibration ──")
    logits_all, labels_all = [], []
    eps = 1e-7
    for imgs, lbls in val_ds:
        preds = model.predict(imgs, verbose=0)
        logits_all.append(np.log(np.clip(preds, eps, 1-eps)))
        labels_all.append(np.argmax(lbls.numpy(), axis=1))
    logits = np.concatenate(logits_all, axis=0)
    labels = np.concatenate(labels_all, axis=0)

    best_T, best_nll = 1.0, float("inf")
    for T in np.arange(0.3, 2.05, 0.05):
        sc   = logits / T
        ex   = np.exp(sc - sc.max(axis=1, keepdims=True))
        prob = ex / ex.sum(axis=1, keepdims=True)
        nll  = -np.mean(np.log(np.clip(prob[np.arange(len(labels)), labels], eps, 1)))
        if nll < best_nll: best_nll = nll; best_T = round(float(T), 2)

    print(f"  Optimal T = {best_T}  (NLL = {best_nll:.4f})")

    def acceptance(logits, T, thresh, margin):
        sc   = logits / T
        ex   = np.exp(sc - sc.max(axis=1, keepdims=True))
        prob = ex / ex.sum(axis=1, keepdims=True)
        top2 = np.sort(prob, axis=1)[:, -2:]
        return ((top2[:,1] >= thresh) & ((top2[:,1]-top2[:,0]) >= margin)).mean()*100

    ar_before = acceptance(logits, 1.0,    CONFIDENCE_THRESHOLD, MIN_TOP2_MARGIN)
    ar_after  = acceptance(logits, best_T, CONFIDENCE_THRESHOLD, MIN_TOP2_MARGIN)
    print(f"  Acceptance before calibration: {ar_before:.1f}%")
    print(f"  Acceptance after  calibration: {ar_after:.1f}%")
    print(f"  Improvement: +{ar_after-ar_before:.1f} pp")
    np.save(str(OUTPUT_DIR/"temperature.npy"), np.array([best_T]))
    return best_T


# ─────────────────────────────────────────────
#  STEP 7 — EVALUATE
# ─────────────────────────────────────────────
def apply_temperature(probs_np, T):
    eps    = 1e-7
    logits = np.log(np.clip(probs_np, eps, 1-eps)) / T
    ex     = np.exp(logits - logits.max(axis=1, keepdims=True))
    return ex / ex.sum(axis=1, keepdims=True)

def evaluate(model, val_ds, history, temperature=1.0):
    print(f"\n── Evaluating (T={temperature}) ──")
    y_true, y_pred = [], []
    per_class_accept = {c: [] for c in CLASS_NAMES}

    for imgs, lbls in val_ds:
        raw   = model.predict(imgs, verbose=0)
        preds = apply_temperature(raw, temperature) if temperature != 1.0 else raw
        tidx  = np.argmax(lbls.numpy(), axis=1)
        pidx  = np.argmax(preds, axis=1)
        y_true.extend(tidx); y_pred.extend(pidx)
        for ti, prob_row in zip(tidx, preds):
            top2   = np.sort(prob_row)[-2:]
            accept = (top2[1] >= CONFIDENCE_THRESHOLD) and \
                     ((top2[1]-top2[0]) >= MIN_TOP2_MARGIN)
            per_class_accept[CLASS_NAMES[ti]].append(accept)

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    print("\nPer-class confusion (true → predicted errors):")
    cm = confusion_matrix(y_true, y_pred)
    for i, tc in enumerate(CLASS_NAMES):
        for j, pc in enumerate(CLASS_NAMES):
            if i != j and cm[i][j] > 0:
                print(f"  {tc:20s} → {pc:20s}: {cm[i][j]} errors")

    print("\nPer-class acceptance:")
    for cls, acc in per_class_accept.items():
        if acc:
            rate = sum(acc)/len(acc)*100
            print(f"  {cls:20s}: {rate:5.1f}%  ({sum(acc)}/{len(acc)})")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    axes[0].set_title(f"Confusion Matrix (T={temperature})")
    axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    ep = range(1, len(history["accuracy"])+1)
    axes[1].plot(ep, history["accuracy"],     label="Train acc")
    axes[1].plot(ep, history["val_accuracy"], label="Val acc")
    axes[1].plot(ep, history["loss"],         "--", label="Train loss")
    axes[1].plot(ep, history["val_loss"],     "--", label="Val loss")
    axes[1].axvline(EPOCHS_FROZEN, color="gray",   linestyle=":", label="Warm fine-tune")
    axes[1].axvline(EPOCHS_FROZEN + EPOCHS_FINETUNE_WARM, color="orange",
                    linestyle=":", label="Full fine-tune")
    axes[1].set_title("Training History"); axes[1].set_xlabel("Epoch")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"training_report.png", dpi=150)
    print(f"\nReport → {OUTPUT_DIR/'training_report.png'}")


# ─────────────────────────────────────────────
#  STEP 8 — TFLITE EXPORT
# ─────────────────────────────────────────────
def export_tflite(model, train_ds):
    print("\n── TFLite INT8 export ──")
    def rep():
        for imgs, _ in train_ds.take(50):
            for img in imgs: yield [tf.expand_dims(img, 0)]
    c = tf.lite.TFLiteConverter.from_keras_model(model)
    c.optimizations                = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset       = rep
    c.target_spec.supported_ops    = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type         = tf.uint8
    c.inference_output_type        = tf.uint8
    out = OUTPUT_DIR / "smartbin_cnn_quant.tflite"
    out.write_bytes(c.convert())
    print(f"TFLite → {out}  ({out.stat().st_size/1024:.1f} KB)")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("="*60)
    print("  Smart AI Bin — Optimised CNN Training v3.1")
    print(f"  TensorFlow {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"  GPU: {gpus or 'None (CPU — will be slow)'}")
    print("="*60)
    download_dataset()
    remap_dataset()
    model, train_ds, val_ds, history = train()
    save_inference_config()
    T = calibrate_temperature(model, val_ds)
    evaluate(model, val_ds, history, T)
    export_tflite(model, train_ds)
    print("\n"+"="*60)
    print(f"  Model : {OUTPUT_DIR/'smartbin_cnn_final.h5'}")
    print(f"  TFLite: {OUTPUT_DIR/'smartbin_cnn_quant.tflite'}")
    print(f"  Temp  : {OUTPUT_DIR/'temperature.npy'}  (T={T})")
    print(f"  Config: {MODEL_CONFIG_PATH}")
    print(f"  Report: {OUTPUT_DIR/'training_report.png'}")
    print("="*60)