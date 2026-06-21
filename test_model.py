"""
Smart AI Bin — Optimised Test Script v3
========================================
Changes from v2:

  FIX 1 — Tighter plastic-vs-glass ambiguity thresholds.
    GLASS_PLASTIC_AMBIG_GAP raised 0.16 → 0.22
    GLASS_PLASTIC_AMBIG_MIN lowered 0.25 → 0.18
    Effect: triggers the ambiguity guard on more transparent-
    plastic-as-glass cases (e.g. 72% glass / 28% plastic now
    flags as ambiguous instead of ACCEPT).

  FIX 2 — New PLASTIC_MIN_CONF_TO_BLOCK_GLASS threshold.
    If predicted class is glass but plastic probability is
    above 0.22, glass is rejected UNLESS confidence is above
    0.80 AND margin above 0.35. Catches transparent PET bottles.

  FIX 3 — Tighter metal-vs-glass guard (silver cans → glass).
    METAL_IF_GLASS_PRESENT lowered 0.20 → 0.15
    METAL_MIN_CONF_WITH_GLASS raised 0.75 → 0.82
    Effect: when there is ANY meaningful glass signal (15%+),
    metal must be very confident (82%+) to be accepted.

  FIX 4 — New GLASS_MIN_CONF_STANDALONE threshold.
    Glass predictions with no plastic or metal evidence still
    need 0.72+ confidence (raised from implicit 0.70) because
    dark glass bottles genuinely look like metal in low light.

  FIX 5 — Smarter recapture → actuator logic.
    Old: "accepted on any single attempt → fire"
    New: "same class on ≥ 2/3 attempts AND best conf ≥ 0.60
          AND no active ambiguity guard → FIRE actuator"
    This prevents misfiring on genuinely ambiguous items that
    happen to agree twice by chance.
    Actuator decision is printed as a separate line:
      ACTUATOR: FIRE  or  ACTUATOR: HOLD (reason)

  FIX 6 — Per-attempt debug output in verbose mode.
    Pass --verbose to see each attempt's probs and guard reasons.
"""

import argparse, json, os, pathlib, platform, subprocess, sys
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras

MODEL_PATH           = "smartbin_project/output/smartbin_cnn_final.h5"
TEMP_PATH            = "smartbin_project/output/temperature.npy"
MODEL_CONFIG_PATH    = "smartbin_project/output/model_config.json"
REMAP_DIR            = pathlib.Path("smartbin_project/dataset")
IMG_SIZE             = 224
PREPROCESS_NAME      = "mobilenet_v2"
CONFIDENCE_THRESHOLD = 0.66
MIN_TOP2_MARGIN      = 0.10
TTA_PASSES           = 1

# ── Ambiguity guards ──────────────────────────────────────────────────────────
# Glass vs metal (solid dark bottles look like metal, silver cans look like glass)
GLASS_METAL_AMBIG_GAP   = 0.18   # unchanged — gap must be >18% to trust prediction
GLASS_AMBIG_MIN         = 0.20   # unchanged
METAL_AMBIG_MIN         = 0.20   # unchanged

# Metal guard: lowered trigger threshold + raised required confidence
# Previously: glass_p >= 0.20 triggered, metal needed 0.75
# Now:        glass_p >= 0.15 triggers, metal needs 0.80
# Catches silver cans that produce 15-18% glass signal
METAL_IF_GLASS_PRESENT      = 0.15   # was 0.20
METAL_MIN_CONF_WITH_GLASS   = 0.80   # was 0.82

# Glass standalone guard: dark glass bottles sometimes confident at 70-72%
# but are actually metal. Raise the floor slightly.
GLASS_MIN_CONF_STANDALONE   = 0.72   # new — applied when plastic+metal both < 0.15

# Plastic vs glass (transparent PET bottles → classified as glass)
# Wider gap required + lower trigger threshold = more rejections on close calls
GLASS_PLASTIC_AMBIG_GAP     = 0.22   # was 0.16 — gap must be >22% to trust glass
GLASS_PLASTIC_AMBIG_MIN     = 0.18   # was 0.25 — triggers at lower plastic evidence

# New: if plastic evidence is above this level, glass needs strong confidence
PLASTIC_MIN_CONF_TO_BLOCK_GLASS  = 0.22   # new threshold
GLASS_MIN_CONF_WITH_PLASTIC_HARD = 0.80   # was 0.75 — raised for hard cases
GLASS_MIN_MARGIN_WITH_PLASTIC_HARD = 0.35 # was 0.30 — raised

# Existing thresholds (kept for softer cases)
GLASS_IF_PLASTIC_PRESENT    = 0.30
GLASS_MIN_MARGIN_WITH_PLASTIC = 0.30
GLASS_MIN_CONF_WITH_PLASTIC   = 0.75

# ── Recapture / fallback ──────────────────────────────────────────────────────
RECAPTURE_ATTEMPTS    = 3
FALLBACK_CONF_FLOOR   = 0.50
FALLBACK_MARGIN_FLOOR = 0.20
FALLBACK_STABLE_RATIO = 0.67
FALLBACK_METAL_MIN_CONF_WITH_GLASS   = 0.80   # matches new METAL_MIN_CONF_WITH_GLASS
FALLBACK_METAL_MIN_MARGIN_WITH_GLASS = 0.35
FALLBACK_GLASS_MIN_MARGIN_WITH_PLASTIC = 0.35  # raised from 0.30
FALLBACK_GLASS_MIN_CONF_WITH_PLASTIC   = 0.80  # raised from 0.75

# ── Actuator logic ────────────────────────────────────────────────────────────
# Fire the actuator when:
#   1. Same class on >= ACTUATOR_MAJORITY_RATIO of attempts
#   2. Best confidence for that class >= ACTUATOR_MIN_CONF
#   3. No active ambiguity guard on the best attempt
ACTUATOR_MAJORITY_RATIO = 0.67   # 2/3 attempts minimum
ACTUATOR_MIN_CONF       = 0.60   # best-attempt confidence floor

GLASS_CLASS        = "glass_nonrecyclable"
LEGACY_GLASS_CLASS = "glass_other"
CLASS_NAMES        = [GLASS_CLASS, "metal", "plastic_paper"]
NUM_CLASSES        = len(CLASS_NAMES)
GLASS_IDX          = CLASS_NAMES.index(GLASS_CLASS)
METAL_IDX          = CLASS_NAMES.index("metal")
PLASTIC_IDX        = CLASS_NAMES.index("plastic_paper")
LABEL_ALIASES      = {LEGACY_GLASS_CLASS: GLASS_CLASS}

CLASS_CMD = {
    GLASS_CLASS     : "G -> Glass/Non-recyclable bin",
    "metal"         : "M -> Metal bin",
    "plastic_paper" : "P -> Plastic/Paper bin",
}
CLASS_COLOR_BGR = {
    GLASS_CLASS     : (0, 165, 255),
    "metal"         : (235, 120, 30),
    "plastic_paper" : (0, 180, 60),
}

PREVIEW_OUT_DIR = pathlib.Path("smartbin_project/output/previews")
VERBOSE = False   # set by --verbose flag


def _is_wsl_env():
    if os.environ.get("WSL_DISTRO_NAME"): return True
    return "microsoft" in platform.release().lower()

def _can_open_gui_windows():
    return not _is_wsl_env()

def _to_windows_path(p):
    p = str(pathlib.Path(p).resolve())
    if p.startswith("/mnt/") and len(p) > 6:
        return f"{p[5].upper()}:{p[6:].replace('/', chr(92))}"
    return p

def _open_in_system_viewer(image_path):
    try:
        if os.name == "nt": os.startfile(str(image_path)); return True
        if _is_wsl_env():
            subprocess.run(["cmd.exe", "/c", "start", "", _to_windows_path(image_path)], check=False)
            return True
        if sys.platform == "darwin": subprocess.run(["open", str(image_path)], check=False); return True
        subprocess.run(["xdg-open", str(image_path)], check=False); return True
    except Exception: return False

def _save_and_open_preview(img_bgr, class_name, conf, probs, T, source_name):
    PREVIEW_OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = pathlib.Path(source_name).stem if source_name else "preview"
    out_path = PREVIEW_OUT_DIR / f"{stem}_overlay.jpg"
    cv2.imwrite(str(out_path), draw_overlay(img_bgr, class_name, conf, probs, T))
    opened = _open_in_system_viewer(out_path)
    print(f"{'Opened' if opened else 'Saved'} preview: {out_path}")

def canonical_label(label):
    return LABEL_ALIASES.get(label, label)

def load_model_and_temperature():
    global PREPROCESS_NAME, IMG_SIZE
    mp = pathlib.Path(MODEL_PATH)
    if not mp.exists():
        print(f"Model not found: {MODEL_PATH}\nRun train_model.py first.")
        sys.exit(1)
    cfg_path = pathlib.Path(MODEL_CONFIG_PATH)
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            PREPROCESS_NAME = str(cfg.get("preprocess", PREPROCESS_NAME))
            IMG_SIZE = int(cfg.get("img_size", IMG_SIZE))
            print(f"Config loaded: preprocess={PREPROCESS_NAME}, img_size={IMG_SIZE}")
        except Exception as e:
            print(f"Warning: could not parse {MODEL_CONFIG_PATH}: {e}")
    print(f"Loading model: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH, compile=False)
    T = 1.0
    tp = pathlib.Path(TEMP_PATH)
    if tp.exists():
        T = float(np.load(str(tp))[0])
        print(f"Temperature T = {T:.2f} loaded")
    else:
        print("temperature.npy not found, using T=1.0")
    print()
    return model, T

def preprocess(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    if PREPROCESS_NAME == "efficientnet_v2":
        img = keras.applications.efficientnet_v2.preprocess_input(img)
    else:
        img = keras.applications.mobilenet_v2.preprocess_input(img)
    return img[np.newaxis, ...]

def apply_temperature(probs, T):
    if T == 1.0: return probs
    eps = 1e-7
    logits = np.log(np.clip(probs, eps, 1-eps)) / T
    ex = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return ex / ex.sum(axis=-1, keepdims=True)

def _augment(img_bgr):
    out = img_bgr.copy()
    h, w = out.shape[:2]
    M = np.float32([[1,0,np.random.randint(-8,9)],[0,1,np.random.randint(-8,9)]])
    out = cv2.warpAffine(out, M, (w,h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=1.0+np.random.uniform(-0.08,0.08),
                              beta=np.random.uniform(-12,12))
    if np.random.rand() < 0.35: out = cv2.GaussianBlur(out, (3,3), 0)
    return out

def predict(model, img_bgr, T=1.0, tta_passes=1):
    tta_passes = max(1, tta_passes)
    probs_list = []
    for i in range(tta_passes):
        sample = img_bgr if i == 0 else _augment(img_bgr)
        raw    = model.predict(preprocess(sample), verbose=0)[0]
        cal    = apply_temperature(raw[np.newaxis,:], T)[0]
        probs_list.append(cal)
    probs = np.mean(np.stack(probs_list, axis=0), axis=0)
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs


def is_accepted(probs, pred_class=None, threshold=None, margin=None):
    """
    Returns (accepted, top1, top2, reasons).
    All three confusion pairs are checked in priority order:
      1. transparent plastic → glass  (tightest guard)
      2. silver metal → glass         (new lower trigger)
      3. dark glass → metal           (existing guard)
    """
    threshold = threshold or CONFIDENCE_THRESHOLD
    margin    = margin    or MIN_TOP2_MARGIN
    top2_vals = np.sort(probs)[-2:]
    t1, t2 = float(top2_vals[1]), float(top2_vals[0])
    accepted = (t1 >= threshold) and ((t1 - t2) >= margin)
    reasons  = []

    glass_p   = float(probs[GLASS_IDX])
    metal_p   = float(probs[METAL_IDX])
    plastic_p = float(probs[PLASTIC_IDX])

    # ── Guard 1: transparent plastic predicted as glass ──────────────────────
    # Trigger: glass predicted AND plastic evidence >= GLASS_PLASTIC_AMBIG_MIN
    # Hard block: plastic >= PLASTIC_MIN_CONF_TO_BLOCK_GLASS and glass not strong enough
    if pred_class == GLASS_CLASS:

        if plastic_p >= PLASTIC_MIN_CONF_TO_BLOCK_GLASS:
            if glass_p < GLASS_MIN_CONF_WITH_PLASTIC_HARD or \
               (t1 - t2) < GLASS_MIN_MARGIN_WITH_PLASTIC_HARD:
                accepted = False
                reasons.append(
                    f"transparent-plastic guard: plastic={plastic_p:.0%} "
                    f"glass_conf={glass_p:.0%}<{GLASS_MIN_CONF_WITH_PLASTIC_HARD:.0%} "
                    f"or margin={(t1-t2):.0%}<{GLASS_MIN_MARGIN_WITH_PLASTIC_HARD:.0%}"
                )

        elif plastic_p >= GLASS_IF_PLASTIC_PRESENT:
            if (glass_p - plastic_p) < GLASS_MIN_MARGIN_WITH_PLASTIC:
                accepted = False
                reasons.append(
                    f"glass~plastic margin too low: "
                    f"margin={glass_p-plastic_p:.0%}<{GLASS_MIN_MARGIN_WITH_PLASTIC:.0%}"
                )
            if glass_p < GLASS_MIN_CONF_WITH_PLASTIC:
                accepted = False
                reasons.append(
                    f"glass<{GLASS_MIN_CONF_WITH_PLASTIC:.0%} with plastic evidence"
                )

        # Gap-based ambiguity (catches very close calls)
        gp_gap = glass_p - plastic_p
        if plastic_p >= GLASS_PLASTIC_AMBIG_MIN and gp_gap < GLASS_PLASTIC_AMBIG_GAP:
            accepted = False
            reasons.append(
                f"glass~plastic ambiguous: gap={gp_gap:.0%}<{GLASS_PLASTIC_AMBIG_GAP:.0%} "
                f"with plastic={plastic_p:.0%}>={GLASS_PLASTIC_AMBIG_MIN:.0%}"
            )

        # Standalone glass guard: even without plastic, very dark glass
        # can be misread — require slight extra confidence
        if plastic_p < 0.15 and metal_p < 0.15 and glass_p < GLASS_MIN_CONF_STANDALONE:
            accepted = False
            reasons.append(
                f"glass standalone conf {glass_p:.0%}<{GLASS_MIN_CONF_STANDALONE:.0%}"
            )

    # ── Guard 2: silver metal can predicted as glass ──────────────────────────
    # Trigger: metal predicted AND glass evidence >= METAL_IF_GLASS_PRESENT (now 0.15)
    if pred_class == "metal" and glass_p >= METAL_IF_GLASS_PRESENT:
        if metal_p < METAL_MIN_CONF_WITH_GLASS:
            accepted = False
            reasons.append(
                f"metal-glass guard: metal={metal_p:.0%}<{METAL_MIN_CONF_WITH_GLASS:.0%} "
                f"with glass={glass_p:.0%}>={METAL_IF_GLASS_PRESENT:.0%}"
            )

    # ── Guard 3: glass~metal gap (both directions) ────────────────────────────
    gm_gap = abs(glass_p - metal_p)
    if gm_gap < GLASS_METAL_AMBIG_GAP:
        if glass_p >= GLASS_AMBIG_MIN and metal_p >= METAL_AMBIG_MIN:
            accepted = False
            reasons.append(
                f"glass~metal ambiguous: gap={gm_gap:.0%}<{GLASS_METAL_AMBIG_GAP:.0%}"
            )

    return accepted, t1, t2, reasons


def _is_glass_metal_ambiguous(probs):
    glass_p = float(probs[GLASS_IDX])
    metal_p = float(probs[METAL_IDX])
    return (abs(glass_p - metal_p) < GLASS_METAL_AMBIG_GAP
            and glass_p >= GLASS_AMBIG_MIN and metal_p >= METAL_AMBIG_MIN)

def _is_glass_plastic_ambiguous(probs):
    glass_p   = float(probs[GLASS_IDX])
    plastic_p = float(probs[PLASTIC_IDX])
    return (glass_p - plastic_p < GLASS_PLASTIC_AMBIG_GAP
            and plastic_p >= GLASS_PLASTIC_AMBIG_MIN)

def _no_active_ambiguity(probs, pred_class):
    """Returns True if no ambiguity guard is currently triggered."""
    _, _, _, reasons = is_accepted(probs, pred_class=pred_class)
    return len(reasons) == 0


def decide_with_recaptures(model, img_bgr, T, tta):
    """
    Runs up to RECAPTURE_ATTEMPTS predictions.
    Returns: (class, conf, probs, accepted, decision_label, reasons, n_attempts, actuator_fire)

    Actuator decision (NEW):
      Fire if: same class on >= ACTUATOR_MAJORITY_RATIO of attempts
               AND best confidence >= ACTUATOR_MIN_CONF
               AND no active ambiguity guard on best attempt
    """
    attempts = []
    n = max(1, RECAPTURE_ATTEMPTS)

    for attempt_i in range(n):
        cls, conf, probs = predict(model, img_bgr, T, tta)
        accepted, t1, t2, reasons = is_accepted(probs, pred_class=cls)
        attempts.append((cls, conf, probs, accepted, t1, t2, reasons))
        if VERBOSE:
            print(f"  [attempt {attempt_i+1}] {cls} conf={conf:.1%} "
                  f"margin={(t1-t2):.1%} accepted={accepted} "
                  f"{'reasons: '+str(reasons) if reasons else ''}")
        if accepted:
            actuator = _evaluate_actuator(attempts)
            return cls, conf, probs, True, "ACCEPT", reasons, len(attempts), actuator

    # ── Fallback logic (unchanged structure, updated thresholds) ──────────────
    counts = {}
    for cls, *_ in attempts:
        counts[cls] = counts.get(cls, 0) + 1
    stable_class = max(counts.items(), key=lambda x: x[1])[0]
    stable_count = counts[stable_class]
    stable_ratio = stable_count / len(attempts)
    class_flips  = len(counts) > 1

    stable_attempts = [a for a in attempts if a[0] == stable_class]
    chosen = max(stable_attempts, key=lambda a: a[1])
    cls, conf, probs, _, t1, t2, _ = chosen

    glass_p   = float(probs[GLASS_IDX])
    metal_p   = float(probs[METAL_IDX])
    plastic_p = float(probs[PLASTIC_IDX])

    fallback_reasons = []
    if stable_ratio < FALLBACK_STABLE_RATIO:
        fallback_reasons.append(f"unstable<{FALLBACK_STABLE_RATIO:.0%}")
    if class_flips:
        fallback_reasons.append("class flip across recaptures")
    if t1 < FALLBACK_CONF_FLOOR:
        fallback_reasons.append(f"top1<{FALLBACK_CONF_FLOOR:.0%}")
    if (t1 - t2) < FALLBACK_MARGIN_FLOOR:
        fallback_reasons.append(f"margin<{FALLBACK_MARGIN_FLOOR:.0%}")
    if _is_glass_metal_ambiguous(probs):
        fallback_reasons.append("glass~metal ambiguous")
    if _is_glass_plastic_ambiguous(probs):
        fallback_reasons.append("glass~plastic ambiguous")

    if cls == "metal" and glass_p >= METAL_IF_GLASS_PRESENT:
        if metal_p < FALLBACK_METAL_MIN_CONF_WITH_GLASS:
            fallback_reasons.append(
                f"metal<{FALLBACK_METAL_MIN_CONF_WITH_GLASS:.0%} with glass evidence")
        if (t1 - t2) < FALLBACK_METAL_MIN_MARGIN_WITH_GLASS:
            fallback_reasons.append(
                f"margin<{FALLBACK_METAL_MIN_MARGIN_WITH_GLASS:.0%} with glass evidence")

    if cls == GLASS_CLASS and plastic_p >= GLASS_IF_PLASTIC_PRESENT:
        if (t1 - t2) < FALLBACK_GLASS_MIN_MARGIN_WITH_PLASTIC:
            fallback_reasons.append(
                f"margin<{FALLBACK_GLASS_MIN_MARGIN_WITH_PLASTIC:.0%} with plastic evidence")
        if t1 < FALLBACK_GLASS_MIN_CONF_WITH_PLASTIC:
            fallback_reasons.append(
                f"top1<{FALLBACK_GLASS_MIN_CONF_WITH_PLASTIC:.0%} with plastic evidence")

    fallback_accept = len(fallback_reasons) == 0
    actuator = _evaluate_actuator(attempts) if fallback_accept else False

    if fallback_accept:
        return (cls, conf, probs, True, "FALLBACK_ACCEPT",
                [f"stable={stable_count}/{len(attempts)}"], len(attempts), actuator)
    return (cls, conf, probs, False, "RECAPTURE",
            fallback_reasons, len(attempts), False)


def _evaluate_actuator(attempts):
    """
    Decides whether to fire the physical actuator.

    Rules:
      1. Same predicted class on >= ACTUATOR_MAJORITY_RATIO of attempts
      2. Best confidence for that class >= ACTUATOR_MIN_CONF
      3. No active ambiguity guard fires on the best attempt for that class

    This prevents the actuator firing when a misconfident coin-flip happens
    to land the same way twice (e.g. transparent bottle at 71% glass x2).
    """
    if not attempts: return False

    counts = {}
    for cls, *_ in attempts:
        counts[cls] = counts.get(cls, 0) + 1

    dominant_cls   = max(counts, key=counts.__getitem__)
    dominant_count = counts[dominant_cls]
    ratio          = dominant_count / len(attempts)

    if ratio < ACTUATOR_MAJORITY_RATIO:
        return False

    # Find the best-confidence attempt for the dominant class
    best = max((a for a in attempts if a[0] == dominant_cls), key=lambda a: a[1])
    best_cls, best_conf, best_probs = best[0], best[1], best[2]

    if best_conf < ACTUATOR_MIN_CONF:
        return False

    if not _no_active_ambiguity(best_probs, best_cls):
        return False

    return True


def print_result(class_name, conf, probs, T, path="", decision_label=None,
                 accepted_override=None, reasons_override=None,
                 attempts_used=1, actuator_fire=False):
    accepted, t1, t2, reasons = is_accepted(probs, pred_class=class_name)
    if accepted_override is not None: accepted = accepted_override
    if reasons_override  is not None: reasons  = reasons_override
    if path: print(f"  File   : {path}")
    print(f"  Result : {class_name}")
    print(f"  Conf   : {conf:.1%}  (T={T:.2f})")
    print(f"  Serial : {CLASS_CMD[class_name]}")
    flag       = decision_label or ("ACCEPT" if accepted else "RECAPTURE")
    reason_txt = f" [{', '.join(reasons)}]" if reasons else ""
    print(f"  Gate   : top1={t1:.1%} top2={t2:.1%} margin={(t1-t2):.1%} -> {flag}{reason_txt}")
    if attempts_used > 1: print(f"  Tries  : {attempts_used}")

    # ── Actuator line ──────────────────────────────────────────────────────────
    if actuator_fire:
        print(f"  ACTUATOR: FIRE  [{class_name} confirmed, conf={conf:.1%}]")
    elif accepted or decision_label in ("ACCEPT","FALLBACK_ACCEPT"):
        print(f"  ACTUATOR: HOLD  [not enough evidence yet]")
    else:
        print(f"  ACTUATOR: HOLD  [ambiguous — re-present item]")

    print()
    print("  Probabilities:")
    for i, name in enumerate(CLASS_NAMES):
        bar = "X" * int(probs[i] * 28)
        print(f"    {name:20s} {probs[i]:5.1%}  {bar}")
    print("-"*50)


def draw_overlay(img_bgr, class_name, conf, probs, T):
    out    = img_bgr.copy()
    color  = CLASS_COLOR_BGR[class_name]
    h, w   = out.shape[:2]
    accepted, _, _, _ = is_accepted(probs, pred_class=class_name)
    cv2.rectangle(out, (0,0), (w,55), (20,20,20), -1)
    cv2.putText(out, f"{class_name}  {conf:.1%}  T={T:.2f}",
                (12,28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(out, CLASS_CMD[class_name],
                (12,50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    bw = max(1, w // NUM_CLASSES)
    for i, name in enumerate(CLASS_NAMES):
        bx   = i * bw
        fill = int(probs[i] * 60)
        cv2.rectangle(out, (bx,h-60), (bx+bw-2,h), (40,40,40), -1)
        cv2.rectangle(out, (bx,h-fill), (bx+bw-2,h), CLASS_COLOR_BGR[name], -1)
        cv2.putText(out, f"{probs[i]:.0%}", (bx+4,h-64),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
        cv2.putText(out, name[:8], (bx+4,h-50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.33, (150,150,150), 1)
    border = color if accepted else (60,60,200)
    cv2.rectangle(out, (0,0), (w-1,h-1), border, 3)
    return out


def test_single(model, T, image_path, tta):
    img = cv2.imread(str(image_path))
    if img is None: print(f"Cannot read: {image_path}"); return
    cls, conf, probs, accepted, decision_label, reasons, tries, actuator = \
        decide_with_recaptures(model, img, T, tta)
    print_result(cls, conf, probs, T, path=str(image_path),
                 decision_label=decision_label, accepted_override=accepted,
                 reasons_override=reasons, attempts_used=tries,
                 actuator_fire=actuator)
    if not _can_open_gui_windows():
        print("GUI preview disabled; opening via system image viewer.")
        _save_and_open_preview(img, cls, conf, probs, T, str(image_path))
        return
    try:
        cv2.imshow("Smart AI Bin", draw_overlay(img, cls, conf, probs, T))
        print("Press any key to close.")
        cv2.waitKey(0); cv2.destroyAllWindows()
    except cv2.error:
        _save_and_open_preview(img, cls, conf, probs, T, str(image_path))


def test_folder(model, T, folder_path, tta, show=False):
    folder = pathlib.Path(folder_path)
    images = [p for ext in ("*.jpg","*.jpeg","*.png") for p in folder.glob(ext)]
    print(f"Found {len(images)} images\n")
    opened_windows = []
    for img_path in sorted(images):
        img = cv2.imread(str(img_path))
        if img is None: continue
        cls, conf, probs, accepted, decision_label, reasons, tries, actuator = \
            decide_with_recaptures(model, img, T, tta)
        print_result(cls, conf, probs, T, path=img_path.name,
                     decision_label=decision_label, accepted_override=accepted,
                     reasons_override=reasons, attempts_used=tries,
                     actuator_fire=actuator)
        if show:
            if not _can_open_gui_windows():
                _save_and_open_preview(img, cls, conf, probs, T, img_path.name)
                continue
            try:
                win_name = f"Smart AI Bin - {img_path.name}"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.imshow(win_name, draw_overlay(img, cls, conf, probs, T))
                cv2.waitKey(1)
                opened_windows.append(win_name)
            except cv2.error:
                _save_and_open_preview(img, cls, conf, probs, T, img_path.name)
    if show and _can_open_gui_windows() and opened_windows:
        print("All previews opened in separate windows.")
        print("Close each image window manually. Press 'q' to stop waiting.")
        try:
            while True:
                visible = [
                    w for w in opened_windows
                    if cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) >= 1
                ]
                if not visible:
                    break
                if cv2.waitKey(200) & 0xFF == ord("q"):
                    break
        except cv2.error:
            pass


def test_webcam(model, T, tta):
    print("Webcam running.  SPACE=classify  Q=quit\n")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("Cannot open webcam."); return
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, "SPACE=classify  Q=quit",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.imshow("Smart AI Bin - Live", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            cls, conf, probs, accepted, decision_label, reasons, tries, actuator = \
                decide_with_recaptures(model, frame, T, tta)
            print_result(cls, conf, probs, T, path="[webcam]",
                         decision_label=decision_label, accepted_override=accepted,
                         reasons_override=reasons, attempts_used=tries,
                         actuator_fire=actuator)
            cv2.imshow("Smart AI Bin - Result",
                       draw_overlay(frame, cls, conf, probs, T))
        elif key == ord("q"): break
    cap.release(); cv2.destroyAllWindows()


def test_val_set(model, T, tta):
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns; import matplotlib.pyplot as plt

    print("Running full val set...\n")
    val_root = REMAP_DIR / "val"
    if not val_root.exists():
        print(f"Val folder not found: {val_root}"); return

    y_true, y_pred, y_accept = [], [], []
    per_class = {c: {"accept":0,"total":0} for c in CLASS_NAMES}
    actuator_fires = 0

    for cls_dir in val_root.iterdir():
        true_label = canonical_label(cls_dir.name)
        if true_label not in CLASS_NAMES: continue
        true_idx = CLASS_NAMES.index(true_label)
        imgs = [p for ext in ("*.jpg","*.jpeg","*.png") for p in cls_dir.glob(ext)]
        for img_path in imgs:
            img = cv2.imread(str(img_path))
            if img is None: continue
            cls, conf, probs, accepted, _, _, _, actuator = \
                decide_with_recaptures(model, img, T, tta)
            y_true.append(true_idx); y_pred.append(CLASS_NAMES.index(cls))
            y_accept.append(accepted)
            per_class[true_label]["total"] += 1
            if accepted: per_class[true_label]["accept"] += 1
            if actuator: actuator_fires += 1

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
    total_acc = sum(y_accept)
    print(f"Overall acceptance (T={T:.2f}, thresh={CONFIDENCE_THRESHOLD:.0%}, "
          f"margin={MIN_TOP2_MARGIN:.0%}):")
    print(f"  {total_acc}/{len(y_accept)} = {total_acc/len(y_accept)*100:.1f}%\n")
    print(f"  Actuator fires: {actuator_fires}/{len(y_accept)} "
          f"= {actuator_fires/len(y_accept)*100:.1f}%\n")
    print("  Per class acceptance:")
    for cls, d in per_class.items():
        if d["total"] > 0:
            pct = d["accept"]/d["total"]*100
            print(f"    {cls:20s}: {pct:5.1f}%  ({d['accept']}/{d['total']})")

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f"Confusion Matrix (T={T:.2f})")
    plt.tight_layout()
    out = "smartbin_project/output/test_confusion.png"
    plt.savefig(out, dpi=150); print(f"\nConfusion matrix -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image",   type=str)
    group.add_argument("--folder",  type=str)
    group.add_argument("--webcam",  action="store_true")
    group.add_argument("--val",     action="store_true")
    parser.add_argument("--threshold",   type=float, default=CONFIDENCE_THRESHOLD)
    parser.add_argument("--margin",      type=float, default=MIN_TOP2_MARGIN)
    parser.add_argument("--tta",         type=int,   default=TTA_PASSES)
    parser.add_argument("--no-tta",      action="store_true")
    parser.add_argument("--recaptures",  type=int,   default=RECAPTURE_ATTEMPTS)
    parser.add_argument("--no-temp",     action="store_true")
    parser.add_argument("--show",        action="store_true")
    parser.add_argument("--verbose",     action="store_true",
                        help="Print per-attempt probabilities and guard reasons.")
    parser.add_argument("--actuator-majority", type=float, default=ACTUATOR_MAJORITY_RATIO,
                        help="Fraction of attempts that must agree to fire actuator (default 0.67).")
    parser.add_argument("--actuator-min-conf", type=float, default=ACTUATOR_MIN_CONF,
                        help="Minimum confidence on best attempt to fire actuator (default 0.60).")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD   = args.threshold
    MIN_TOP2_MARGIN        = args.margin
    tta_passes             = 1 if args.no_tta else args.tta
    RECAPTURE_ATTEMPTS     = max(1, args.recaptures)
    ACTUATOR_MAJORITY_RATIO = args.actuator_majority
    ACTUATOR_MIN_CONF       = args.actuator_min_conf
    VERBOSE                = args.verbose

    model, T = load_model_and_temperature()
    if args.no_temp: T = 1.0; print("Temperature scaling disabled (T=1.0)\n")

    if args.image:    test_single(model, T, args.image, tta_passes)
    elif args.folder: test_folder(model, T, args.folder, tta_passes, show=args.show)
    elif args.webcam: test_webcam(model, T, tta_passes)
    elif args.val:    test_val_set(model, T, tta_passes)