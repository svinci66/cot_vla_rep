#!/bin/bash

# Two-GPU Phase 4 Visual CoT smoke-test entrypoint.
# Verifies the paper-aligned path where visual loss trains the depth
# transformer while DDP keeps trainable parameters synchronized.

set -e

REPO_ROOT=${REPO_ROOT:-"/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep"}
CONDA_PREFIX_PATH=${CONDA_PREFIX_PATH:-"/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed"}

cd "$REPO_ROOT"
export PATH="$CONDA_PREFIX_PATH/bin:$PATH"

# Server/cache defaults.
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-"vila_env_fixed"}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}
export REPORT_TO=${REPORT_TO:-none}
export WANDB_DISABLED=${WANDB_DISABLED:-true}
export HF_HOME=${HF_HOME:-"/data/share/1919650160032350208/sj/hf_cache_shared"}
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"}
export DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-cot-libero-goal-2gpu-phase4-smoke"}

# Reduce noisy startup/link warnings. Errors are still shown.
export QUIET_TRAINING_LOGS=${QUIET_TRAINING_LOGS:-True}
if [ "$QUIET_TRAINING_LOGS" = "True" ] || [ "$QUIET_TRAINING_LOGS" = "true" ]; then
    export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
    export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
    export HF_HUB_VERBOSITY=${HF_HUB_VERBOSITY:-error}
    export TQDM_DISABLE=${TQDM_DISABLE:-1}
    export WANDB_SILENT=${WANDB_SILENT:-true}
    export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore::FutureWarning,ignore::UserWarning}
fi

# Two-GPU smoke-test defaults. Override these env vars for larger runs.
export SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-False}
export NUM_GPUS=${NUM_GPUS:-2}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export NUM_EPOCHS=${NUM_EPOCHS:-1}
export BATCH_SIZE=${BATCH_SIZE:-4}
export ACC_STEP=${ACC_STEP:-1}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
export SAVE_STRATEGY=${SAVE_STRATEGY:-no}
export SAVE_STEPS=${SAVE_STEPS:-100}
export LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS=${LIGHTWEIGHT_EVAL_CHECKPOINT_EPOCHS:-0}
export MAX_DEMOS_PER_TASK=${MAX_DEMOS_PER_TASK:-50}
export MASTER_PORT=${MASTER_PORT:-25007}

# Phase 4 Visual CoT defaults.
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_VISUAL_COT_LOSS=${USE_VISUAL_COT_LOSS:-True}
export TUNE_DEPTH_TRANSFORMER=${TUNE_DEPTH_TRANSFORMER:-True}
export TUNE_VISION_TOWER=${TUNE_VISION_TOWER:-False}
export TUNE_LANGUAGE_MODEL=${TUNE_LANGUAGE_MODEL:-True}
export TUNE_MM_PROJECTOR=${TUNE_MM_PROJECTOR:-True}
export VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-1.0}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}
export XYZ_LOSS_WEIGHT=${XYZ_LOSS_WEIGHT:-1.5}
export GRIPPER_CLOSE_LOSS_WEIGHT=${GRIPPER_CLOSE_LOSS_WEIGHT:-2.0}
export GRIPPER_TRANSITION_LOSS_WEIGHT=${GRIPPER_TRANSITION_LOSS_WEIGHT:-4.0}
export USE_ACTION_PERCENTILE_BINS=${USE_ACTION_PERCENTILE_BINS:-True}
export ACTION_BIN_LOW_PERCENTILE=${ACTION_BIN_LOW_PERCENTILE:-1.0}
export ACTION_BIN_HIGH_PERCENTILE=${ACTION_BIN_HIGH_PERCENTILE:-99.0}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-1}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-${ACTION_CHUNK_SIZE:-10}}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-uniform}

# DDP diagnostics for the smoke test.
export RANK_SLICE_AFTER_SHUFFLE=${RANK_SLICE_AFTER_SHUFFLE:-True}
export SAMPLER_DEBUG=${SAMPLER_DEBUG:-True}
export RANK_PARAMETER_CHECK=${RANK_PARAMETER_CHECK:-True}

# Hybrid attention currently requires eager 4D masks.
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
export LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-True}
export USE_DEEPSPEED=${USE_DEEPSPEED:-False}
export SYNC_TRANSFORMERS_PATCH=${SYNC_TRANSFORMERS_PATCH:-True}

bash scripts/train/train_action_prediction.sh "$@"
