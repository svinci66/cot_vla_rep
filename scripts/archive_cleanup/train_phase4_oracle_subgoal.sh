#!/bin/bash

# Phase 4 oracle-subgoal action baseline.
# Reuses Phase 3 discrete action training and hybrid attention, while sampling
# future video frames online as GT subgoal images.

export OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase4-oracle-subgoal"}
export USE_VISUAL_COT=${USE_VISUAL_COT:-True}
export USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
export SUBGOAL_MIN_OFFSET=${SUBGOAL_MIN_OFFSET:-1}
export SUBGOAL_MAX_OFFSET=${SUBGOAL_MAX_OFFSET:-${ACTION_CHUNK_SIZE:-10}}
export SUBGOAL_SAMPLING_STRATEGY=${SUBGOAL_SAMPLING_STRATEGY:-uniform}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/train/train_action_prediction.sh" "$@"
