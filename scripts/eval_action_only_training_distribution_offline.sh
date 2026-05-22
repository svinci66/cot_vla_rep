#!/bin/bash

# Strict offline action eval for action-only checkpoints.
# This uses the same LiberoGoalDataset + discrete collator + hybrid-attention
# forward path as training, instead of raw HDF5 timestep scanning.

MODEL_PATH=${MODEL_PATH:-"./checkpoints/vila-u-action-only-overfit-1task-10demo-paper-aligned"}
DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
DEVICE=${DEVICE:-cuda}
MODEL_DTYPE=${MODEL_DTYPE:-bfloat16}
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_SAMPLES=${MAX_SAMPLES:-500}
NUM_WORKERS=${NUM_WORKERS:-0}
IMAGE_SIZE=${IMAGE_SIZE:-256}
REMOVE_PAUSE_INTERVALS=${REMOVE_PAUSE_INTERVALS:-True}
PAUSE_THRESHOLD=${PAUSE_THRESHOLD:-0.01}
GRIPPER_PAUSE_THRESHOLD=${GRIPPER_PAUSE_THRESHOLD:-1e-6}
MAX_TASK_FILES=${MAX_TASK_FILES:-1}
MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-1}
TASK_FILE=${TASK_FILE:-}
TASK_FILE_PATTERN=${TASK_FILE_PATTERN:-}
OUTPUT_JSON=${OUTPUT_JSON:-"outputs/action_only_training_distribution_offline_eval.json"}
SAVE_RECORDS=${SAVE_RECORDS:-False}
RECOMPUTE_ACTION_BIN_EDGES=${RECOMPUTE_ACTION_BIN_EDGES:-False}

cmd=(
    python scripts/eval_action_training_distribution_offline.py
    --model-path "$MODEL_PATH"
    --data-root "$DATA_ROOT"
    --device "$DEVICE"
    --model-dtype "$MODEL_DTYPE"
    --batch-size "$BATCH_SIZE"
    --max-samples "$MAX_SAMPLES"
    --num-workers "$NUM_WORKERS"
    --image-size "$IMAGE_SIZE"
    --remove-pause-intervals "$REMOVE_PAUSE_INTERVALS"
    --pause-threshold "$PAUSE_THRESHOLD"
    --gripper-pause-threshold "$GRIPPER_PAUSE_THRESHOLD"
    --output-json "$OUTPUT_JSON"
)

if [ -n "$MAX_TASK_FILES" ]; then
    cmd+=(--max-task-files "$MAX_TASK_FILES")
fi

if [ -n "$MAX_DEMOS_PER_TASK" ]; then
    cmd+=(--max-demos-per-task "$MAX_DEMOS_PER_TASK")
fi

if [ -n "$TASK_FILE" ]; then
    cmd+=(--task-file "$TASK_FILE")
fi

if [ -n "$TASK_FILE_PATTERN" ]; then
    cmd+=(--task-file-pattern "$TASK_FILE_PATTERN")
fi

if [ "$SAVE_RECORDS" = "True" ] || [ "$SAVE_RECORDS" = "true" ]; then
    cmd+=(--save-records)
fi

if [ "$RECOMPUTE_ACTION_BIN_EDGES" = "True" ] || [ "$RECOMPUTE_ACTION_BIN_EDGES" = "true" ]; then
    cmd+=(--recompute-action-bin-edges)
fi

echo "Running strict training-distribution offline action eval"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_ROOT"
echo "  Task File: ${TASK_FILE:-auto}"
echo "  Task Pattern: ${TASK_FILE_PATTERN:-none}"
echo "  Max Task Files: ${MAX_TASK_FILES:-none}"
echo "  Max Demos Per Task: ${MAX_DEMOS_PER_TASK:-none}"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Remove Pause Intervals: $REMOVE_PAUSE_INTERVALS"
echo "  Output: $OUTPUT_JSON"

"${cmd[@]}"
