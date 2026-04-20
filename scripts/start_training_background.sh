#!/bin/bash

# ==========================================
# VILA-U 8-GPU Training Launcher (Background)
# ==========================================

# ==========================================
# 1. 加载 Conda 配置
# ==========================================
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
# 4. 拉取最新代码
# ==========================================
echo ">>> Pulling latest code..."
git fetch origin
git pull origin phase3-flash-attention

# ==========================================
# 5. 启动训练任务（后台运行）
# ==========================================
LOG_FILE="/data/private/training_$(date +%Y%m%d_%H%M%S).log"
echo ">>> Starting training in background..."
echo ">>> Config: 8 GPUs, Batch Size 160"
echo ">>> Log file: $LOG_FILE"

# 导出环境变量
export SINGLE_GPU_MODE=False
export NUM_GPUS=8
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export BATCH_SIZE=160

# 后台运行训练
nohup bash scripts/train/train_action_prediction.sh > "$LOG_FILE" 2>&1 &

# 获取进程ID
TRAIN_PID=$!
echo ">>> Training started with PID: $TRAIN_PID"
echo ">>> To monitor: tail -f $LOG_FILE"
echo ">>> To check status: ps -p $TRAIN_PID"
echo ">>> To stop: kill $TRAIN_PID"

# 保存PID到文件
echo "$TRAIN_PID" > /data/private/training.pid
echo ">>> PID saved to /data/private/training.pid"
