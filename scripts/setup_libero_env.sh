#!/usr/bin/env bash
set -euo pipefail

# Create a standalone LIBERO environment for online simulator checks.
# Keep this separate from the VILA-U training environment because LIBERO's
# official stack pins an older Python/PyTorch/robosuite combination.

ENV_NAME="${LIBERO_ENV_NAME:-libero}"
ENV_PREFIX="${LIBERO_ENV_PREFIX:-}"
LIBERO_ROOT="${LIBERO_ROOT:-$HOME/repo/LIBERO}"
DATASETS="${LIBERO_DATASETS:-}"
USE_HUGGINGFACE="${LIBERO_USE_HUGGINGFACE:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install miniconda/anaconda first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if [ -n "$ENV_PREFIX" ]; then
    if [ ! -d "$ENV_PREFIX" ]; then
        mkdir -p "$(dirname "$ENV_PREFIX")"
        conda create -p "$ENV_PREFIX" python=3.8.13 -y
    fi
    conda activate "$ENV_PREFIX"
    ENV_DISPLAY="$ENV_PREFIX"
    ACTIVATE_CMD="conda activate $ENV_PREFIX"
else
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -n "$ENV_NAME" python=3.8.13 -y
    fi
    conda activate "$ENV_NAME"
    ENV_DISPLAY="$ENV_NAME"
    ACTIVATE_CMD="conda activate $ENV_NAME"
fi

echo "Using LIBERO conda environment: $ENV_DISPLAY"

python -m pip install --upgrade pip setuptools wheel

if [ ! -d "$LIBERO_ROOT/.git" ]; then
    mkdir -p "$(dirname "$LIBERO_ROOT")"
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi

cd "$LIBERO_ROOT"

python -m pip install -r requirements.txt
python -m pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
python -m pip install robosuite
python -m pip install -e .

if [ -n "$DATASETS" ]; then
    if [ "$USE_HUGGINGFACE" = "1" ]; then
        python benchmark_scripts/download_libero_datasets.py --datasets "$DATASETS" --use-huggingface
    else
        python benchmark_scripts/download_libero_datasets.py --datasets "$DATASETS"
    fi
fi

python - <<'PY'
import libero
import robosuite
print("LIBERO import ok:", getattr(libero, "__file__", None))
print("robosuite import ok:", getattr(robosuite, "__version__", "unknown"))
PY

cat <<EOF

LIBERO environment is ready.

Next:
  $ACTIVATE_CMD
  export MUJOCO_GL=egl
  export MUJOCO_EGL_DEVICE_ID=0
  python $REPO_ROOT/scripts/check_libero_online.py --suite libero_goal --task-id 0 --steps 10
EOF
