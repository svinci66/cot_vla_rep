#!/bin/bash

# Single-GPU action-only overfit debug run.
# Uses 1 LIBERO task file and 5-10 demos to verify action loss can collapse.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-only-overfit-1task-10demo"}

export SINGLE_GPU_MODE=True
export NUM_GPUS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export USE_VISUAL_COT=False
export USE_VISUAL_COT_LOSS=False
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}

export TUNE_LANGUAGE_MODEL=True
export TUNE_MM_PROJECTOR=True
export TUNE_DEPTH_TRANSFORMER=True
export TUNE_VISION_TOWER=False

export MAX_TASK_FILES=${MAX_TASK_FILES:-1}
export MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-10}
export TASK_FILE=${TASK_FILE:-}
export TASK_FILE_PATTERN=${TASK_FILE_PATTERN:-}

export NUM_EPOCHS=${NUM_EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-4}
export ACC_STEP=${ACC_STEP:-1}
export LEARNING_RATE=${LEARNING_RATE:-1e-5}
export SAVE_STEPS=${SAVE_STEPS:-200}
export SAVE_STRATEGY=${SAVE_STRATEGY:-no}
export LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS=${LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS:-0}
export LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCH_LIST=${LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCH_LIST:-10,12,15}
export SAVE_ONLY_TRAINABLE=${SAVE_ONLY_TRAINABLE:-True}
export REPORT_TO=${REPORT_TO:-none}
export WANDB_DISABLED=${WANDB_DISABLED:-true}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train/train_action_prediction.sh" "$@"
