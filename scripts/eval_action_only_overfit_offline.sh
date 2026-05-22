#!/bin/bash

# Offline action eval for the single-task action-only overfit checkpoint.
# Defaults to the strict training-distribution eval path.

MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep/checkpoints/vila-u-action-only-overfit-1task-10demo-fresh-20260519_180135"}
DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
DEVICE=${DEVICE:-cuda}
MAX_SAMPLES=${MAX_SAMPLES:-2000}
MAX_TASK_FILES=${MAX_TASK_FILES:-1}
MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-10}
TASK_FILE=${TASK_FILE:-}
TASK_FILE_PATTERN=${TASK_FILE_PATTERN:-}
OUTPUT_JSON=${OUTPUT_JSON:-"outputs/action_only_overfit_offline_eval.json"}
SAVE_RECORDS=${SAVE_RECORDS:-False}
STRICT_TRAINING_DISTRIBUTION=${STRICT_TRAINING_DISTRIBUTION:-True}

if [ "$STRICT_TRAINING_DISTRIBUTION" = "True" ] || [ "$STRICT_TRAINING_DISTRIBUTION" = "true" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    export MODEL_PATH DATA_ROOT DEVICE MAX_SAMPLES MAX_TASK_FILES MAX_DEMOS_PER_TASK
    export TASK_FILE TASK_FILE_PATTERN OUTPUT_JSON SAVE_RECORDS
    exec "$SCRIPT_DIR/eval_action_only_training_distribution_offline.sh"
fi

STRIDE=${STRIDE:-1}

cmd=(
    python scripts/eval_phase3_actions_offline.py
    --model-path "$MODEL_PATH"
    --data-root "$DATA_ROOT"
    --device "$DEVICE"
    --max-samples "$MAX_SAMPLES"
    --stride "$STRIDE"
    --max-task-files "$MAX_TASK_FILES"
    --max-demos-per-task "$MAX_DEMOS_PER_TASK"
    --output-json "$OUTPUT_JSON"
)

if [ -n "$TASK_FILE" ]; then
    cmd+=(--task-file "$TASK_FILE")
fi

if [ -n "$TASK_FILE_PATTERN" ]; then
    cmd+=(--task-file-pattern "$TASK_FILE_PATTERN")
fi

if [ "$SAVE_RECORDS" = "True" ] || [ "$SAVE_RECORDS" = "true" ]; then
    cmd+=(--save-records)
fi

echo "Running offline action eval"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_ROOT"
echo "  Task File: ${TASK_FILE:-auto}"
echo "  Task Pattern: ${TASK_FILE_PATTERN:-none}"
echo "  Max Task Files: $MAX_TASK_FILES"
echo "  Max Demos Per Task: $MAX_DEMOS_PER_TASK"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Output: $OUTPUT_JSON"

"${cmd[@]}"
