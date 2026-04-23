#!/bin/bash
# Phase 4: Visual CoT-VLA Training Script (Simplified)
#
# Usage:
#   bash scripts/train/train_cot_vla_simple.sh
#
# Required environment variables (or edit below):
#   MODEL_PATH: Path to pretrained VILA-U model
#   DATA_ROOT: Path to LIBERO Goal dataset

set -e

# ===== Required Configuration (EDIT THESE) =====
MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"}
DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/phase4_visual_cot"}

# ===== Optional Configuration =====
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
BATCH_SIZE=${BATCH_SIZE:-4}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-2}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-500}
LOGGING_STEPS=${LOGGING_STEPS:-10}

# ===== Print Configuration =====
echo "=========================================="
echo "Phase 4: Visual CoT-VLA Training"
echo "=========================================="
echo "Model: $MODEL_PATH"
echo "Data: $DATA_ROOT"
echo "Output: $OUTPUT_DIR"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Batch size: $BATCH_SIZE"
echo "Gradient accumulation: $GRADIENT_ACCUMULATION_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "=========================================="

# ===== Run Training =====
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python -m vila_u.train.train_cot_vla \
    --model_name_or_path "$MODEL_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --version v1 \
    --mm_projector mlp2x_gelu \
    --tune_mm_projector True \
    --tune_language_model True \
    --tune_vision_tower False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_use_vi_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio resize \
    --bf16 True \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps $SAVE_STEPS \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps $LOGGING_STEPS \
    --tf32 True \
    --model_max_length 1536 \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="
