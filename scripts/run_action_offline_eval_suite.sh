#!/bin/bash

set -euo pipefail

if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    MODEL_PATH_ARG="$1"
    shift
else
    MODEL_PATH_ARG="${MODEL_PATH:-}"
fi
if [ -z "$MODEL_PATH_ARG" ]; then
    echo "Usage: $0 /path/to/checkpoint [extra run_action_offline_eval_suite.py args...]" >&2
    echo "Or set MODEL_PATH=/path/to/checkpoint and run $0 [extra args...]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

VILA_ENV_BIN=${VILA_ENV_BIN:-"/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed/bin"}
if [ -d "$VILA_ENV_BIN" ]; then
    export PATH="$VILA_ENV_BIN:$PATH"
fi

export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export HF_HOME=${HF_HOME:-"/data/share/1919650160032350208/sj/hf_cache_shared"}
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

exec python scripts/run_action_offline_eval_suite.py \
    --model-path "$MODEL_PATH_ARG" \
    "$@"
