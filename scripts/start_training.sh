#!/bin/bash

# ==========================================
# VILA-U 8-GPU Training Launcher
# ==========================================

# ==========================================
# 1. 加载 Conda 配置
# ==========================================
# 使用 source 而不是直接运行，以确保环境变量在当前 shell 中生效
if [ -f /data/private/miniconda.sh ]; then
    echo ">>> Sourcing miniconda.sh..."
    source /data/private/miniconda.sh
else
    echo ">>> Error: /data/private/miniconda.sh not found!"
    exit 1
fi

# ==========================================
# 2. 激活 Conda 环境
# ==========================================
ENV_PATH="/data/share/1919650160032350208/sj/conda_pkgs/vila_env_fixed"
echo ">>> Activating conda environment: $ENV_PATH"
conda activate "$ENV_PATH"

# 检查环境是否激活成功
if [ $? -ne 0 ]; then
    echo ">>> Error: Failed to activate conda environment."
    exit 1
fi

echo ">>> Conda environment activated: $(which python)"

# ==========================================
# 3. 进入项目目录
# ==========================================
PROJECT_DIR="/data/share/1919650160032350208/sj/cot-vla/cot_vla_rep"
echo ">>> Changing directory to: $PROJECT_DIR"
cd "$PROJECT_DIR"

if [ $? -ne 0 ]; then
    echo ">>> Error: Failed to change directory."
    exit 1
fi

echo ">>> Current directory: $(pwd)"

# ==========================================
# 4. 拉取最新代码（可选）
# ==========================================
echo ">>> Checking git status..."
git fetch origin
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/phase3-flash-attention)

if [ "$LOCAL" != "$REMOTE" ]; then
    echo ">>> Local is behind remote. Pulling latest code..."
    git pull origin phase3-flash-attention
else
    echo ">>> Already up to date."
fi

# ==========================================
# 5. 启动训练任务
# ==========================================
echo ">>> Starting training..."
echo ">>> Config: 8 GPUs, Batch Size 160"
echo ">>> Log file: /data/private/training.log"

# 导出环境变量
export SINGLE_GPU_MODE=False
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_SIZE=160

# 启动训练（注意：bash命令在最后）
bash scripts/train/train_action_prediction.sh

echo ">>> Training started!"
