#!/bin/bash

# 8-GPU Phase 4 Visual CoT action prediction training wrapper.
# This is intentionally separate from legacy train_action_prediction_8.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_ROOT"

# Environment defaults for the current server layout.
export CONDA_ENV_NAME=${CONDA_ENV_NAME:-"vila_env_fixed"}
export MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"}
export DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase4-visual-cot-8gpu"}
export HF_HOME=${HF_HOME:-"/data/share/1919650160032350208/sj/hf_cache_shared"}
export HF_ENDPOINT=${HF_ENDPOINT:-"https://hf-mirror.com"}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

# Distributed defaults. BATCH_SIZE is global batch size; the shared script will
# derive per-device batch as BATCH_SIZE / (NUM_GPUS * ACC_STEP).
export SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-False}
export NUM_GPUS=${NUM_GPUS:-8}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export BATCH_SIZE=${BATCH_SIZE:-160}
export ACC_STEP=${ACC_STEP:-10}
export MASTER_PORT=${MASTER_PORT:-25001}

# Phase 4 Visual CoT defaults.
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_VISUAL_COT_LOSS=${USE_VISUAL_COT_LOSS:-True}
export VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-1.0}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-1}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-${ACTION_CHUNK_SIZE:-10}}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-uniform}

# Safe speed/stability defaults for current hybrid-attention implementation.
export GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-True}
export DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-8}
export ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-eager}
export LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-True}
export USE_DEEPSPEED=${USE_DEEPSPEED:-False}
export SYNC_TRANSFORMERS_PATCH=${SYNC_TRANSFORMERS_PATCH:-True}

exec "$SCRIPT_DIR/train_action_prediction.sh" "$@"
