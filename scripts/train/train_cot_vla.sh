#!/bin/bash
# Phase 4: Visual CoT-VLA Training Script
#
# This script trains the CoT-VLA model with visual chain-of-thought reasoning:
# observation -> subgoal image generation -> action prediction

set -e

# ===== Configuration =====
MODEL_PATH=${MODEL_PATH:-"mit-han-lab/vila-u-7b-256"}
DATA_ROOT=${DATA_ROOT:-"/path/to/libero_goal"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/phase4_visual_cot"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-""}  # Optional: resume from checkpoint

# Training hyperparameters
BATCH_SIZE=${BATCH_SIZE:-4}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_EPOCHS=${NUM_EPOCHS:-10}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-4}
WARMUP_STEPS=${WARMUP_STEPS:-100}

# Phase 4 specific parameters
USE_VISUAL_COT=true
SUBGOAL_HORIZON_LOW=4
SUBGOAL_HORIZON_HIGH=16

# Action prediction parameters (from Phase 2/3)
USE_DISCRETE_ACTION_PREDICTION=true
USE_HYBRID_ATTENTION=true
ACTION_CHUNK_SIZE=10
ACTION_DIM=7
ACTION_NUM_BINS=256

# Training settings
NUM_WORKERS=${NUM_WORKERS:-4}
IMAGE_SIZE=256
USE_WANDB=${USE_WANDB:-false}

# ===== Print Configuration =====
echo "=========================================="
echo "Phase 4: Visual CoT-VLA Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "Epochs: $NUM_EPOCHS"
echo "Visual CoT: $USE_VISUAL_COT"
echo "Subgoal horizon: [$SUBGOAL_HORIZON_LOW, $SUBGOAL_HORIZON_HIGH]"
echo "Discrete actions: $USE_DISCRETE_ACTION_PREDICTION"
echo "Hybrid attention: $USE_HYBRID_ATTENTION"
echo "=========================================="

# ===== Create output directory =====
mkdir -p "$OUTPUT_DIR"

# ===== Save configuration =====
cat > "$OUTPUT_DIR/config.json" <<EOF
{
  "phase": 4,
  "phase_name": "visual_cot",
  "model_path": "$MODEL_PATH",
  "data_root": "$DATA_ROOT",
  "batch_size": $BATCH_SIZE,
  "learning_rate": $LEARNING_RATE,
  "num_epochs": $NUM_EPOCHS,
  "gradient_accumulation_steps": $GRADIENT_ACCUMULATION_STEPS,
  "use_visual_cot": $USE_VISUAL_COT,
  "subgoal_horizon_low": $SUBGOAL_HORIZON_LOW,
  "subgoal_horizon_high": $SUBGOAL_HORIZON_HIGH,
  "use_discrete_action_prediction": $USE_DISCRETE_ACTION_PREDICTION,
  "use_hybrid_attention": $USE_HYBRID_ATTENTION,
  "action_chunk_size": $ACTION_CHUNK_SIZE,
  "action_dim": $ACTION_DIM,
  "action_num_bins": $ACTION_NUM_BINS,
  "image_size": $IMAGE_SIZE
}
EOF

# ===== Run training =====
python -m vila_u.train.train_cot_vla \
    --model_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --warmup_steps $WARMUP_STEPS \
    --use_visual_cot \
    --subgoal_horizon_low $SUBGOAL_HORIZON_LOW \
    --subgoal_horizon_high $SUBGOAL_HORIZON_HIGH \
    --use_discrete_action_prediction \
    --use_hybrid_attention \
    --action_chunk_size $ACTION_CHUNK_SIZE \
    --action_dim $ACTION_DIM \
    --action_num_bins $ACTION_NUM_BINS \
    --image_size $IMAGE_SIZE \
    --num_workers $NUM_WORKERS \
    $([ "$USE_WANDB" = "true" ] && echo "--use_wandb") \
    "$@"

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
