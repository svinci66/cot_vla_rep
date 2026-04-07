#!/bin/bash
# VILA-U 动作预测训练脚本示例

# 设置路径
MODEL_PATH="path/to/vila-u-pretrained-model"  # VILA-U 预训练模型路径
DATA_ROOT="path/to/libero_goal_dataset"       # LIBERO Goal 数据集路径
OUTPUT_DIR="./checkpoints/action_prediction"  # 输出目录

# 训练参数
BATCH_SIZE=8
LEARNING_RATE=1e-5
NUM_EPOCHS=10
GRADIENT_ACCUMULATION_STEPS=2

# 运行训练
python -m vila_u.train.train_action_prediction \
    --model_path $MODEL_PATH \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_DIR \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --use_wandb  # 可选：使用 wandb 记录训练过程

# 说明：
# 1. MODEL_PATH: 下载 VILA-U 预训练模型后的路径
# 2. DATA_ROOT: LIBERO Goal 数据集目录，包含 .hdf5 文件
# 3. 训练会冻结视觉编码器，只训练 LLM 和动作预测头
# 4. 检查点会保存到 OUTPUT_DIR
