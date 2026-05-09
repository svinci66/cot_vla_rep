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
PYTHON_VERSION="${LIBERO_PYTHON_VERSION:-3.8.13}"
RECREATE_ENV="${LIBERO_RECREATE_ENV:-0}"
TORCH_VARIANT="${LIBERO_TORCH_VARIANT:-cpu}"
PIP_RETRIES="${LIBERO_PIP_RETRIES:-10}"
PIP_TIMEOUT="${LIBERO_PIP_TIMEOUT:-120}"
CONDA_RETRIES="${LIBERO_CONDA_RETRIES:-5}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

retry() {
    local attempts="$1"
    shift
    local n=1
    until "$@"; do
        if [ "$n" -ge "$attempts" ]; then
            return 1
        fi
        echo "Command failed. Retrying $n/$attempts: $*" >&2
        sleep $((n * 5))
        n=$((n + 1))
    done
}

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install miniconda/anaconda first." >&2
    exit 1
fi

eval "$(conda shell.bash hook)"

if [ -n "$ENV_PREFIX" ]; then
    if [ "$RECREATE_ENV" = "1" ] && [ -d "$ENV_PREFIX" ]; then
        conda env remove -p "$ENV_PREFIX" -y
    fi
    if [ ! -d "$ENV_PREFIX" ]; then
        mkdir -p "$(dirname "$ENV_PREFIX")"
        retry "$CONDA_RETRIES" conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
    fi
    conda activate "$ENV_PREFIX"
    ENV_DISPLAY="$ENV_PREFIX"
    ACTIVATE_CMD="conda activate $ENV_PREFIX"
else
    if [ "$RECREATE_ENV" = "1" ]; then
        conda env remove -n "$ENV_NAME" -y || true
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        retry "$CONDA_RETRIES" conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi
    conda activate "$ENV_NAME"
    ENV_DISPLAY="$ENV_NAME"
    ACTIVATE_CMD="conda activate $ENV_NAME"
fi

echo "Using LIBERO conda environment: $ENV_DISPLAY"
python - <<'PY'
import platform
import sys
impl = platform.python_implementation()
print(f"Python implementation: {impl} {sys.version.split()[0]}")
if impl != "CPython":
    raise SystemExit(
        "LIBERO setup requires CPython. This environment is not CPython; "
        "rerun with LIBERO_RECREATE_ENV=1 or choose a clean LIBERO_ENV_PREFIX."
    )
PY

retry "$CONDA_RETRIES" conda install -c conda-forge cmake ninja "numpy=1.22.4" -y
retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" --upgrade pip setuptools wheel

if [ ! -d "$LIBERO_ROOT/.git" ]; then
    mkdir -p "$(dirname "$LIBERO_ROOT")"
    retry "$CONDA_RETRIES" git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi

cd "$LIBERO_ROOT"

retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" --no-build-isolation -r requirements.txt
if [ "$TORCH_VARIANT" = "cu113" ]; then
    retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
elif [ "$TORCH_VARIANT" = "cpu" ]; then
    retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu
else
    echo "Unsupported LIBERO_TORCH_VARIANT=$TORCH_VARIANT. Use cpu or cu113." >&2
    exit 1
fi
retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" -e .

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
