#!/bin/bash

# Phase 4 Visual CoT training.
# Adds subgoal visual residual-code loss on top of the oracle-subgoal action path.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase4-visual-cot"}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_VISUAL_COT_LOSS=${USE_VISUAL_COT_LOSS:-True}
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export VISUAL_LOSS_WEIGHT=${VISUAL_LOSS_WEIGHT:-1.0}
export ACTION_LOSS_WEIGHT=${ACTION_LOSS_WEIGHT:-1.0}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-1}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-${ACTION_CHUNK_SIZE:-10}}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-uniform}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train/train_action_prediction.sh" "$@"
