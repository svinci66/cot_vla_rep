#!/bin/bash

# Four-GPU action-only single-task multi-demo debug run.
# Keeps the action-only overfit knobs from the 1-GPU wrapper, but launches via torchrun.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-only-1task-50demo-paper-aligned"}

export SINGLE_GPU_MODE=False
export NUM_GPUS=${NUM_GPUS:-4}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}

export USE_VISUAL_COT=False
export USE_VISUAL_COT_LOSS=False
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}

export TUNE_LANGUAGE_MODEL=True
export TUNE_MM_PROJECTOR=True
export TUNE_DEPTH_TRANSFORMER=True
export TUNE_VISION_TOWER=False

export MAX_TASK_FILES=${MAX_TASK_FILES:-1}
export MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-50}
export TASK_FILE=${TASK_FILE:-open_the_middle_drawer_of_the_cabinet_demo.hdf5}
export TASK_FILE_PATTERN=${TASK_FILE_PATTERN:-}

export NUM_EPOCHS=${NUM_EPOCHS:-20}
export BATCH_SIZE=${BATCH_SIZE:-8}
export ACC_STEP=${ACC_STEP:-1}
export LEARNING_RATE=${LEARNING_RATE:-3e-5}
export SAVE_STEPS=${SAVE_STEPS:-500}
export SAVE_STRATEGY=${SAVE_STRATEGY:-no}
export LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS=${LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS:-0}
export LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCH_LIST=${LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCH_LIST:-10,12,15}
export SAVE_ONLY_TRAINABLE=${SAVE_ONLY_TRAINABLE:-True}
export REPORT_TO=${REPORT_TO:-none}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export MASTER_PORT=${MASTER_PORT:-25001}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train/train_action_prediction.sh" "$@"
