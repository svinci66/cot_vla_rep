#!/usr/bin/env bash
set -euo pipefail

# Recreate the VILA-U training environment from manifests exported by
# scripts/export_vila_env_fixed.sh. If manifests are unavailable, this falls
# back to a best-effort install from public package indexes and this repo.

ENV_NAME="${VILA_ENV_NAME:-vila_env_fixed}"
ENV_PREFIX="${VILA_ENV_PREFIX:-}"
SPEC_DIR="${VILA_ENV_SPEC_DIR:-env_specs/vila_env_fixed}"
USE_CONDA_EXPLICIT="${VILA_USE_CONDA_EXPLICIT:-1}"
USE_PIP_FREEZE="${VILA_USE_PIP_FREEZE:-1}"
PYTHON_VERSION="${VILA_PYTHON_VERSION:-3.12}"
TORCH_VERSION="${VILA_TORCH_VERSION:-2.8.0}"
TORCH_INDEX_URL="${VILA_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
FLASH_ATTN_SPEC="${VILA_FLASH_ATTN_SPEC:-}"
INSTALL_FLASH_ATTN="${VILA_INSTALL_FLASH_ATTN:-1}"
PIP_CACHE_DIR="${VILA_PIP_CACHE_DIR:-}"
PIP_RETRIES="${VILA_PIP_RETRIES:-10}"
PIP_TIMEOUT="${VILA_PIP_TIMEOUT:-120}"
CONDA_RETRIES="${VILA_CONDA_RETRIES:-5}"
RECREATE_ENV="${VILA_RECREATE_ENV:-0}"
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

pip_install() {
    retry "$PIP_RETRIES" python -m pip install --retries "$PIP_RETRIES" --timeout "$PIP_TIMEOUT" "$@"
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
    if [ ! -d "$ENV_PREFIX" ] && [ "$USE_CONDA_EXPLICIT" = "1" ] && [ -f "$SPEC_DIR/conda-explicit.txt" ]; then
        mkdir -p "$(dirname "$ENV_PREFIX")"
        retry "$CONDA_RETRIES" conda create -p "$ENV_PREFIX" --file "$SPEC_DIR/conda-explicit.txt" -y
    elif [ ! -d "$ENV_PREFIX" ]; then
        mkdir -p "$(dirname "$ENV_PREFIX")"
        retry "$CONDA_RETRIES" conda create -p "$ENV_PREFIX" "python=$PYTHON_VERSION" -y
    fi
    conda activate "$ENV_PREFIX"
    ACTIVATE_CMD="conda activate $ENV_PREFIX"
else
    if [ "$RECREATE_ENV" = "1" ]; then
        conda env remove -n "$ENV_NAME" -y || true
    fi
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME" && [ "$USE_CONDA_EXPLICIT" = "1" ] && [ -f "$SPEC_DIR/conda-explicit.txt" ]; then
        retry "$CONDA_RETRIES" conda create -n "$ENV_NAME" --file "$SPEC_DIR/conda-explicit.txt" -y
    elif ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        retry "$CONDA_RETRIES" conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
    fi
    conda activate "$ENV_NAME"
    ACTIVATE_CMD="conda activate $ENV_NAME"
fi

if [ -n "$PIP_CACHE_DIR" ]; then
    mkdir -p "$PIP_CACHE_DIR"
    export PIP_CACHE_DIR
fi

cd "$REPO_ROOT"

python - <<'PY'
import platform
import sys
impl = platform.python_implementation()
print(f"Python implementation: {impl} {sys.version.split()[0]}")
if impl != "CPython":
    raise SystemExit("VILA-U setup requires CPython.")
PY

retry "$CONDA_RETRIES" conda install -c conda-forge cmake ninja -y
pip_install --upgrade pip setuptools wheel

if [ "$USE_PIP_FREEZE" = "1" ] && [ -f "$SPEC_DIR/pip-freeze-clean.txt" ]; then
    pip_install -r "$SPEC_DIR/pip-freeze-clean.txt"
elif [ "$USE_PIP_FREEZE" = "1" ] && [ -f "$SPEC_DIR/pip-freeze.txt" ]; then
    pip_install -r "$SPEC_DIR/pip-freeze.txt"
else
    echo "Installing PyTorch $TORCH_VERSION from $TORCH_INDEX_URL"
    pip_install "torch==$TORCH_VERSION" torchvision torchaudio --index-url "$TORCH_INDEX_URL"

    if [ "$INSTALL_FLASH_ATTN" = "1" ]; then
        if [ -n "$FLASH_ATTN_SPEC" ]; then
            pip_install "$FLASH_ATTN_SPEC"
        else
            echo "VILA_FLASH_ATTN_SPEC is empty; trying source flash-attn install."
            echo "Set VILA_INSTALL_FLASH_ATTN=0 to skip, or VILA_FLASH_ATTN_SPEC to a wheel URL/path for faster setup."
            pip_install flash-attn --no-build-isolation
        fi
    fi

    PATCHED_PYPROJECT="$(mktemp -t vila_pyproject.XXXXXX)"
    cp pyproject.toml "$PATCHED_PYPROJECT"
    restore_pyproject() {
        cp "$PATCHED_PYPROJECT" pyproject.toml
        rm -f "$PATCHED_PYPROJECT"
    }
    trap restore_pyproject EXIT

    python - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
replacements = {
    '"torch==2.3.0"': '"torch>=2.8.0"',
    '"torchvision==0.18.0"': '"torchvision>=0.19.0"',
    '"accelerate==0.34.2"': '"accelerate>=0.34.2"',
    '"bitsandbytes==0.41.0"': '"bitsandbytes>=0.41.0"',
    '"numpy==1.26.4"': '"numpy>=1.26.4,<2.0"',
    '"sentencepiece==0.1.99"': '"sentencepiece>=0.1.99"',
    '"datasets==2.16.1"': '"datasets>=2.19.0"',
}
for old, new in replacements.items():
    text = text.replace(old, new)
path.write_text(text)
PY

    pip_install -e ".[train,eval]"
    pip_install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    pip_install transformers==4.36.2
    pip_install antlr4-python3-runtime==4.9.3

    python -m pip uninstall opencv-python opencv-contrib-python opencv-python-headless -y || true
    pip_install opencv-python-headless
    pip_install "numpy>=1.26.4,<2.0" --force-reinstall
    pip_install --upgrade --force-reinstall --no-deps scikit-learn
    pip_install joblib scipy threadpoolctl
fi

if [ "$INSTALL_FLASH_ATTN" = "1" ] && ! python - <<'PY' >/dev/null 2>&1
import flash_attn
PY
then
    if [ -n "$FLASH_ATTN_SPEC" ]; then
        pip_install "$FLASH_ATTN_SPEC"
    else
        echo "flash-attn is not installed; trying source flash-attn install."
        echo "Set VILA_INSTALL_FLASH_ATTN=0 to skip, or VILA_FLASH_ATTN_SPEC to a wheel URL/path for faster setup."
        pip_install flash-attn --no-build-isolation
    fi
fi

if ! python -m pip show vila-u >/dev/null 2>&1; then
    PATCHED_PYPROJECT="$(mktemp -t vila_pyproject.XXXXXX)"
    cp pyproject.toml "$PATCHED_PYPROJECT"
    restore_pyproject() {
        cp "$PATCHED_PYPROJECT" pyproject.toml
        rm -f "$PATCHED_PYPROJECT"
    }
    trap restore_pyproject EXIT
    python - <<'PY'
from pathlib import Path

path = Path("pyproject.toml")
text = path.read_text()
for old, new in {
    '"torch==2.3.0"': '"torch>=2.8.0"',
    '"torchvision==0.18.0"': '"torchvision>=0.19.0"',
    '"numpy==1.26.4"': '"numpy>=1.26.4,<2.0"',
}.items():
    text = text.replace(old, new)
path.write_text(text)
PY
    pip_install -e ".[train,eval]"
fi

site_pkg_path="$(python -c 'import site; print(site.getsitepackages()[0])')"
cp -rv ./vila_u/train/transformers_replace/* "$site_pkg_path/transformers/"
rm -rf "$site_pkg_path/lmms_eval/models/mplug_owl_video/modeling_mplug_owl.py"

python - <<'PY'
import torch
import transformers
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "available:", torch.cuda.is_available())
print("transformers:", transformers.__version__)
PY

cat <<EOF

VILA-U environment is ready.

Next:
  $ACTIVATE_CMD
  cd $REPO_ROOT
EOF
