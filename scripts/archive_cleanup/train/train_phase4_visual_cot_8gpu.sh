#!/bin/bash

# 8-GPU Phase 4 Visual CoT training entrypoint.
# This script bakes in the server command used for current Phase 4 training.

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
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase4-visual-cot-8gpu"}

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

# 8-GPU launch defaults from the validated command.
export SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-False}
export NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export BATCH_SIZE=${BATCH_SIZE:-160}
export ACC_STEP=${ACC_STEP:-10}
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}

# Phase 4 Visual CoT defaults.
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_VISUAL_COT_LOSS=${USE_VISUAL_COT_LOSS:-True}
export TUNE_DEPTH_TRANSFORMER=${TUNE_DEPTH_TRANSFORMER:-True}
export VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-1.0}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}
export USE_ACTION_PERCENTILE_BINS=${USE_ACTION_PERCENTILE_BINS:-False}
export ACTION_BIN_LOW_PERCENTILE=${ACTION_BIN_LOW_PERCENTILE:-1.0}
export ACTION_BIN_HIGH_PERCENTILE=${ACTION_BIN_HIGH_PERCENTILE:-99.0}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-1}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-${ACTION_CHUNK_SIZE:-10}}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-uniform}

# Hybrid attention currently requires eager 4D masks.
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
export LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-True}
export USE_DEEPSPEED=${USE_DEEPSPEED:-False}
export SYNC_TRANSFORMERS_PATCH=${SYNC_TRANSFORMERS_PATCH:-True}

bash scripts/train/train_action_prediction.sh "$@"
