#!/bin/bash

# Action-only training entrypoint.
# Disables Visual CoT subgoal sampling/loss and trains the action path directly.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-only"}
export USE_VISUAL_COT=False
export USE_VISUAL_COT_LOSS=False
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}

export TUNE_LANGUAGE_MODEL=True
export TUNE_MM_PROJECTOR=True
export TUNE_DEPTH_TRANSFORMER=True
export TUNE_VISION_TOWER=False

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train/train_action_prediction.sh" "$@"
