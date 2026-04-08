#!/bin/bash

# VILA-U Action Prediction Training Script
# Based on VILA-U's sft.sh training framework

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Editable configuration
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"vila_env"}
MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"}
DATA_ROOT=${DATA_ROOT:-"/path/to/libero_goal"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-True}

NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
ACC_STEP=${ACC_STEP:-4}
NUM_EPOCHS=${NUM_EPOCHS:-10}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-500}
MASTER_PORT=${MASTER_PORT:-25001}

IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"resize"}
IMAGE_SIZE=${IMAGE_SIZE:-256}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-10}
ACTION_DIM=${ACTION_DIM:-7}
REMOVE_PAUSE_INTERVALS=${REMOVE_PAUSE_INTERVALS:-True}
PAUSE_THRESHOLD=${PAUSE_THRESHOLD:-0.01}
REPORT_TO=${REPORT_TO:-wandb}
SUPPRESS_FUTURE_WARNING=${SUPPRESS_FUTURE_WARNING:-True}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-True}
USE_DEEPSPEED=${USE_DEEPSPEED:-False}

if [ "$SUPPRESS_FUTURE_WARNING" = "True" ] || [ "$SUPPRESS_FUTURE_WARNING" = "true" ]; then
    export PYTHONWARNINGS="ignore::FutureWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"
fi

export ATTN_IMPLEMENTATION
export LOW_CPU_MEM_USAGE
export CUDA_VISIBLE_DEVICES

# Activate environment if needed
if [ -z "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"  # 根据你的环境名称修改
fi

# SLURM configuration (如果使用 SLURM)
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

if [ "$SINGLE_GPU_MODE" = "True" ] || [ "$SINGLE_GPU_MODE" = "true" ]; then
    NUM_GPUS=1
    n_node=1
    USE_DEEPSPEED=False
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
fi

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: ${SLURM_JOB_ID:-local} | Full list: $worker_list"

# Batch size configuration
global_bs=$BATCH_SIZE
acc_step=$ACC_STEP
if [ "$SINGLE_GPU_MODE" = "True" ] || [ "$SINGLE_GPU_MODE" = "true" ]; then
    if [ "$global_bs" -lt 1 ]; then
        echo "Error: BATCH_SIZE must be >= 1 in single GPU mode" >&2
        exit 1
    fi
    if [ "$acc_step" -gt "$global_bs" ]; then
        acc_step=$global_bs
    fi
    while [ "$acc_step" -gt 1 ] && [ $((global_bs % acc_step)) -ne 0 ]; do
        acc_step=$((acc_step - 1))
    done
fi
batch_divisor=$((n_node * NUM_GPUS * acc_step))
if [ $((global_bs % batch_divisor)) -ne 0 ]; then
    echo "Error: BATCH_SIZE=$global_bs must be divisible by nnodes*NUM_GPUS*ACC_STEP=$batch_divisor" >&2
    exit 1
fi
bs=$((global_bs / batch_divisor))
effective_bs=$((bs * n_node * NUM_GPUS * acc_step))

echo "GLOBAL_BATCH_SIZE="$global_bs
echo "PER_DEVICE_TRAIN_BATCH_SIZE="$bs
echo "GRADIENT_ACCUMULATION_STEPS="$acc_step

echo "=========================================="
echo "Training Configuration:"
echo "  Conda Env: $CONDA_ENV_NAME"
echo "  Model: $MODEL_PATH"
echo "  Data: $DATA_ROOT"
echo "  Output: $OUTPUT_DIR"
echo "  CUDA Visible Devices: $CUDA_VISIBLE_DEVICES"
echo "  Single GPU Mode: $SINGLE_GPU_MODE"
echo "  GPUs per Node: $NUM_GPUS"
echo "  Epochs: $NUM_EPOCHS"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Batch Size: $bs per device"
echo "  Gradient Accumulation: $acc_step"
echo "  Effective Batch Size: $effective_bs"
echo "  Image Size: $IMAGE_SIZE"
echo "  Action Chunk Size: $ACTION_CHUNK_SIZE"
echo "  Suppress FutureWarning: $SUPPRESS_FUTURE_WARNING"
echo "  Attention Backend: $ATTN_IMPLEMENTATION"
echo "  Low CPU Mem Usage: $LOW_CPU_MEM_USAGE"
echo "  Use DeepSpeed: $USE_DEEPSPEED"
echo "=========================================="

# Build training args
train_args=(
    --model_name_or_path "$MODEL_PATH"
    --data_root "$DATA_ROOT"
    --version v1
    --mm_projector mlp2x_gelu
    --tune_mm_projector True
    --tune_language_model True
    --tune_vision_tower False
    --mm_vision_select_layer -2
    --mm_use_im_start_end True
    --mm_use_vi_start_end False
    --mm_use_im_patch_token False
    --image_aspect_ratio "$IMAGE_ASPECT_RATIO"
    --image_size "$IMAGE_SIZE"
    --bf16 True
    --output_dir "$OUTPUT_DIR"
    --num_train_epochs "$NUM_EPOCHS"
    --per_device_train_batch_size "$bs"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "$acc_step"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps "$SAVE_STEPS"
    --save_total_limit 3
    --learning_rate "$LEARNING_RATE"
    --weight_decay 0.
    --warmup_ratio "$WARMUP_RATIO"
    --lr_scheduler_type cosine
    --logging_steps 10
    --tf32 True
    --model_max_length 2048
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --lazy_preprocess True
    --report_to "$REPORT_TO"
    --action_chunk_size "$ACTION_CHUNK_SIZE"
    --action_dim "$ACTION_DIM"
    --remove_pause_intervals "$REMOVE_PAUSE_INTERVALS"
    --pause_threshold "$PAUSE_THRESHOLD"
)

if [ "$USE_DEEPSPEED" = "True" ] || [ "$USE_DEEPSPEED" = "true" ]; then
    train_args+=(--deepspeed ./scripts/zero2.json)
fi

if [ "$SINGLE_GPU_MODE" = "True" ] || [ "$SINGLE_GPU_MODE" = "true" ]; then
    python vila_u/train/train_action_prediction_mem.py "${train_args[@]}"
else
    torchrun --nnodes=$n_node --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        --master_addr $MASTER_ADDR --node_rank=${CURRENT_RANK} \
        vila_u/train/train_action_prediction_mem.py "${train_args[@]}"
fi
