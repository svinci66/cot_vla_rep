#!/bin/bash

# Backward-compatible entrypoint.
# Redirects to the maintained training script under scripts/train/ with
# Phase-3-friendly defaults so old commands do not accidentally resume or reuse
# legacy checkpoints.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase3"}
export SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-True}
export USE_DEEPSPEED=${USE_DEEPSPEED:-False}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

bash scripts/train/train_action_prediction.sh "$@"
