#!/usr/bin/env bash
set -euo pipefail

# Export dependency manifests from an existing VILA-U environment that is
# available on the source server. The output directory can be committed or
# copied to another server and consumed by scripts/setup_vila_env_fixed.sh.

ENV_PREFIX="${VILA_ENV_PREFIX:-/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed}"
OUT_DIR="${VILA_ENV_EXPORT_DIR:-env_specs/vila_env_fixed}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "conda not found. Please install miniconda/anaconda first." >&2
    exit 1
fi

if [ ! -d "$ENV_PREFIX" ]; then
    echo "Environment path not found: $ENV_PREFIX" >&2
    exit 1
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_PREFIX"

cd "$REPO_ROOT"
mkdir -p "$OUT_DIR"
export VILA_ENV_EXPORT_DIR_ABS="$OUT_DIR"

conda list --explicit > "$OUT_DIR/conda-explicit.txt"
conda env export --no-builds > "$OUT_DIR/conda-env-no-builds.yml"
python -m pip freeze --all > "$OUT_DIR/pip-freeze.txt"
python - <<'PY'
import os
from pathlib import Path

repo_root = Path.cwd().resolve()
out_dir = Path(os.environ["VILA_ENV_EXPORT_DIR_ABS"])
src = out_dir / "pip-freeze.txt"
dst = out_dir / "pip-freeze-clean.txt"

clean_lines = []
for raw_line in src.read_text().splitlines():
    line = raw_line.strip()
    if not line:
        continue
    if line.startswith("-e "):
        target = line[3:].strip()
        if target.startswith(".") or target.startswith("/") or target.startswith("file://"):
            continue
    if " @ file://" in line:
        continue
    if str(repo_root) in line:
        continue
    clean_lines.append(raw_line)

dst.write_text("\n".join(clean_lines) + "\n")
PY

python - <<'PY' > "$OUT_DIR/env-info.txt"
import json
import platform
import sys

info = {
    "python": sys.version,
    "implementation": platform.python_implementation(),
    "platform": platform.platform(),
}
try:
    import torch
    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["torch_cuda_available"] = torch.cuda.is_available()
except Exception as exc:
    info["torch_error"] = repr(exc)
try:
    import transformers
    info["transformers"] = transformers.__version__
except Exception as exc:
    info["transformers_error"] = repr(exc)
try:
    import flash_attn
    info["flash_attn"] = getattr(flash_attn, "__version__", "unknown")
except Exception as exc:
    info["flash_attn_error"] = repr(exc)

print(json.dumps(info, indent=2, sort_keys=True))
PY

cat > "$OUT_DIR/README.md" <<EOF
# VILA-U Environment Export

This directory was generated from:

\`\`\`
$ENV_PREFIX
\`\`\`

Files:

- \`conda-explicit.txt\`: exact conda package URLs for the source platform.
- \`conda-env-no-builds.yml\`: portable conda environment export without build strings.
- \`pip-freeze.txt\`: pip packages from the source environment.
- \`pip-freeze-clean.txt\`: pip packages with local editable/file paths removed.
- \`env-info.txt\`: Python, torch, CUDA, transformers, and flash-attn metadata.

On a target server, run:

\`\`\`bash
export VILA_ENV_SPEC_DIR=$OUT_DIR
export VILA_ENV_PREFIX=/path/to/new/vila_env_fixed
bash scripts/setup_vila_env_fixed.sh
\`\`\`

If \`conda-explicit.txt\` contains source-server-only package URLs or is not
portable to the target server, set:

\`\`\`bash
export VILA_USE_CONDA_EXPLICIT=0
\`\`\`
EOF

echo "Exported VILA-U environment manifests to $OUT_DIR"
