#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV="${SORTIFY_VENV:-$HOME/venvs/sortify}"
export SMARTBIN_BACKBONE=${SMARTBIN_BACKBONE:-mobilenet_v2}
export SMARTBIN_FIT_VERBOSE=${SMARTBIN_FIT_VERBOSE:-2}
export LD_LIBRARY_PATH="$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VENV/lib/python3.12/site-packages/nvidia/curand/lib:$VENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.12/site-packages/nvidia/nccl/lib:$VENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"

exec "$VENV/bin/python" train_model.py
