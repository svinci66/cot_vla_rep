#!/bin/bash

# Minimal oracle-subgoal action finetune smoke.
# Uses GT t+10 future images as action-conditioning inputs and disables
# visual-token generation loss.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT=${REPO_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
cd "$REPO_ROOT"

export CONDA_ENV_NAME=${CONDA_ENV_NAME:-"vila_env_fixed"}
export MODEL_PATH=${MODEL_PATH:-"/tmp/cot_vla_eval_6_16_alltask_ckp60"}
export DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/smoke-oracle-subgoal-action-task1-2demo-1ep"}

export SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-True}
export NUM_GPUS=${NUM_GPUS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

export TASK_FILE=${TASK_FILE:-"put_the_bowl_on_the_stove_demo.hdf5"}
export MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-2}
export NUM_EPOCHS=${NUM_EPOCHS:-1}

export ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-10}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-10}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-10}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-fixed}

export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_VISUAL_COT_LOSS=${USE_VISUAL_COT_LOSS:-False}
export VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-0.0}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}

export TUNE_LANGUAGE_MODEL=${TUNE_LANGUAGE_MODEL:-False}
export TUNE_MM_PROJECTOR=${TUNE_MM_PROJECTOR:-True}
export TUNE_VISION_TOWER=${TUNE_VISION_TOWER:-False}
export TUNE_DEPTH_TRANSFORMER=${TUNE_DEPTH_TRANSFORMER:-False}

export SAVE_STRATEGY=${SAVE_STRATEGY:-no}
export SAVE_ONLY_TRAINABLE=${SAVE_ONLY_TRAINABLE:-True}
export AUTO_NEW_OUTPUT_DIR=${AUTO_NEW_OUTPUT_DIR:-True}
export REPORT_TO=${REPORT_TO:-none}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-0}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
export USE_DEEPSPEED=${USE_DEEPSPEED:-False}
export SYNC_TRANSFORMERS_PATCH=${SYNC_TRANSFORMERS_PATCH:-True}

# Keep the action-token semantics from the baseline checkpoint when available.
export USE_ACTION_PERCENTILE_BINS=${USE_ACTION_PERCENTILE_BINS:-True}

# This wrapper treats BATCH_SIZE as per-device batch size for smoke commands.
# The underlying shared launcher expects global batch size.
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-${BATCH_SIZE:-1}}
export ACC_STEP=${ACC_STEP:-${GRADIENT_ACCUMULATION_STEPS:-8}}
export BATCH_SIZE=$((PER_DEVICE_BATCH_SIZE * NUM_GPUS * ACC_STEP))

exec "$SCRIPT_DIR/train_action_prediction.sh" "$@"
