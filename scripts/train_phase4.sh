#!/bin/bash

# Phase 4: Visual CoT-VLA Training Script
# 简化的训练启动脚本

# 设置环境变量
export WANDB_PROJECT="VILA-U-Visual-CoT"
export CUDA_VISIBLE_DEVICES=0
export ATTN_IMPLEMENTATION="eager"  # Phase 4 需要使用 eager attention
export ACCELERATE_DISPATCH_BATCHES="0"  # 禁用 dispatch_batches 以兼容旧版本 accelerate

# 模型和数据路径
MODEL_PATH="/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"
DATA_ROOT="/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"
OUTPUT_DIR="./checkpoints/phase4-visual-cot"

# 训练参数
BATCH_SIZE=4
LEARNING_RATE=2e-5
NUM_EPOCHS=10
MODEL_MAX_LENGTH=1536  # 需要更长的序列长度以容纳子目标 tokens

# Phase 4 特定参数
SUBGOAL_HORIZON=5  # 子目标时间跨度（未来第 5 帧）
VISUAL_LOSS_WEIGHT=1.0
ACTION_LOSS_WEIGHT=1.0

# 启动训练
python -m vila_u.train.train_visual_cot \
    --model_name_or_path $MODEL_PATH \
    --version "llama3" \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps 4 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.01 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 3 \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --bf16 True \
    --tf32 True \
    --mm_use_im_start_end True \
    --mm_use_im_patch_token False \
    --tune_language_model True \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --action_chunk_size 10 \
    --action_dim 7 \
    --image_size 256 \
    --subgoal_horizon $SUBGOAL_HORIZON \
    --remove_pause_intervals True \
    --pause_threshold 0.01 \
    --visual_loss_weight $VISUAL_LOSS_WEIGHT \
    --action_loss_weight $ACTION_LOSS_WEIGHT \
    --report_to "wandb"
