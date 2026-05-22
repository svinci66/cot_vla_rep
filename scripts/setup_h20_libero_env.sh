#!/usr/bin/env bash
set -euo pipefail

# H20 server helper for creating a standalone LIBERO environment under:
#   /data/share/1919650160032350208/sj/cot-vla/
#
# This environment is intended for headless online LIBERO reset/step/render
# checks and ZMQ online evaluation. Keep it separate from the VILA-U model env.

BASE_DIR="${H20_COT_VLA_DIR:-/data/share/1919650160032350208/sj/cot-vla}"
CONDA_BIN="${H20_CONDA_BIN:-/data/share/1919650160032350208/sj/conda_env/miniconda3/bin/conda}"
ENV_PREFIX="${LIBERO_ENV_PREFIX:-$BASE_DIR/conda_envs/libero}"
LIBERO_ROOT="${LIBERO_ROOT:-$BASE_DIR/LIBERO}"
REPO_ROOT="${VILA_REPO_ROOT:-$BASE_DIR/cot_vla_rep}"
DATASETS="${LIBERO_DATASETS:-}"
USE_HUGGINGFACE="${LIBERO_USE_HUGGINGFACE:-0}"
PYTHON_VERSION="${LIBERO_PYTHON_VERSION:-3.8.13}"
RECREATE_ENV="${LIBERO_RECREATE_ENV:-0}"
TORCH_VARIANT="${LIBERO_TORCH_VARIANT:-cpu}"
PIP_RETRIES="${LIBERO_PIP_RETRIES:-10}"
PIP_TIMEOUT="${LIBERO_PIP_TIMEOUT:-120}"
CONDA_RETRIES="${LIBERO_CONDA_RETRIES:-5}"
MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"

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

pip_install() {
    retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "$@"
}

echo "=========================================="
echo "H20 LIBERO headless environment setup"
echo "=========================================="
echo "BASE_DIR: $BASE_DIR"
echo "ENV_PREFIX: $ENV_PREFIX"
echo "LIBERO_ROOT: $LIBERO_ROOT"
echo "REPO_ROOT: $REPO_ROOT"
echo "TORCH_VARIANT: $TORCH_VARIANT"
echo ""

mkdir -p "$BASE_DIR" "$BASE_DIR/conda_envs"

if [ -x "$CONDA_BIN" ]; then
    eval "$("$CONDA_BIN" shell.bash hook)"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    echo "conda not found. Set H20_CONDA_BIN to the conda executable." >&2
    exit 1
fi

if [ "$RECREATE_ENV" = "1" ] && [ -d "$ENV_PREFIX" ]; then
    conda env remove -p "$ENV_PREFIX" -y
fi

if [ ! -d "$ENV_PREFIX" ]; then
    mkdir -p "$(dirname "$ENV_PREFIX")"
    retry "$CONDA_RETRIES" conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_PREFIX"

python - <<'PY'
import platform
import sys
impl = platform.python_implementation()
print(f"Python implementation: {impl} {sys.version.split()[0]}")
if impl != "CPython":
    raise SystemExit(
        "LIBERO setup requires CPython. Rerun with LIBERO_RECREATE_ENV=1 "
        "or choose a clean LIBERO_ENV_PREFIX."
    )
PY

retry "$CONDA_RETRIES" conda install -c conda-forge cmake ninja "numpy=1.22.4" -y
pip_install --upgrade pip setuptools wheel

if [ ! -d "$LIBERO_ROOT/.git" ]; then
    mkdir -p "$(dirname "$LIBERO_ROOT")"
    retry "$CONDA_RETRIES" git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_ROOT"
fi

cd "$LIBERO_ROOT"

pip_install --no-build-isolation -r requirements.txt

case "$TORCH_VARIANT" in
    cpu)
        pip_install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 \
            --extra-index-url https://download.pytorch.org/whl/cpu
        ;;
    cu113)
        pip_install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 \
            --extra-index-url https://download.pytorch.org/whl/cu113
        ;;
    *)
        echo "Unsupported LIBERO_TORCH_VARIANT=$TORCH_VARIANT. Use cpu or cu113." >&2
        exit 1
        ;;
esac

pip_install -e .
pip_install pyzmq msgpack msgpack-numpy imageio imageio-ffmpeg

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

LIBERO environment is ready on H20.

Next:
  conda activate $ENV_PREFIX
  cd $REPO_ROOT
  export MUJOCO_GL=egl
  export MUJOCO_EGL_DEVICE_ID=$MUJOCO_EGL_DEVICE_ID
  python scripts/check_libero_online.py --suite libero_goal --task-id 0 --steps 10

Optional online rollout after the model worker is running:
  python scripts/libero_zmq_env_server.py \\
    --suite libero_goal \\
    --task-id 0 \\
    --episodes 10 \\
    --max-steps 300 \\
    --host 127.0.0.1 \\
    --port 5555 \\
    --output-dir outputs/h20_libero_online
EOF
