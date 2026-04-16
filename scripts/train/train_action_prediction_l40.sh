#!/bin/bash

# VILA-U Action Prediction Training Script for L40
# Conservative single-GPU defaults for easier bring-up on L40

export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# Editable configuration
CONDA_ENV_NAME=${CONDA_ENV_NAME:-"vila_env_fixed"}
MODEL_PATH=${MODEL_PATH:-"/data/share/1919650160032350208/sj/vila-u/vila-u-7b-256"}
DATA_ROOT=${DATA_ROOT:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/vila-u-action-prediction-phase3-l40"}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
SINGLE_GPU_MODE=${SINGLE_GPU_MODE:-True}

NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-20}
ACC_STEP=${ACC_STEP:-2}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SAVE_STEPS=${SAVE_STEPS:-2000}
LOGGING_STEPS=${LOGGING_STEPS:-50}
MASTER_PORT=${MASTER_PORT:-25011}

IMAGE_ASPECT_RATIO=${IMAGE_ASPECT_RATIO:-"resize"}
IMAGE_SIZE=${IMAGE_SIZE:-256}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-512}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
DATALOADER_PREFETCH_FACTOR=${DATALOADER_PREFETCH_FACTOR:-2}
DATALOADER_PERSISTENT_WORKERS=${DATALOADER_PERSISTENT_WORKERS:-True}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-False}
ACTION_CHUNK_SIZE=${ACTION_CHUNK_SIZE:-10}
ACTION_DIM=${ACTION_DIM:-7}
REMOVE_PAUSE_INTERVALS=${REMOVE_PAUSE_INTERVALS:-True}
PAUSE_THRESHOLD=${PAUSE_THRESHOLD:-0.01}
REPORT_TO=${REPORT_TO:-none}
SUPPRESS_FUTURE_WARNING=${SUPPRESS_FUTURE_WARNING:-True}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
LOW_CPU_MEM_USAGE=${LOW_CPU_MEM_USAGE:-True}
USE_DEEPSPEED=${USE_DEEPSPEED:-False}
USE_HYBRID_ATTENTION=${USE_HYBRID_ATTENTION:-True}
ALLOW_FLASH_HYBRID_ATTENTION=${ALLOW_FLASH_HYBRID_ATTENTION:-False}
SYNC_TRANSFORMERS_PATCH=${SYNC_TRANSFORMERS_PATCH:-True}
RESUME_TRAINING=${RESUME_TRAINING:-False}
AUTO_NEW_OUTPUT_DIR=${AUTO_NEW_OUTPUT_DIR:-True}
USE_TORCH_COMPILE=${USE_TORCH_COMPILE:-False}
TORCH_COMPILE_MODE=${TORCH_COMPILE_MODE:-default}

if [ "$SUPPRESS_FUTURE_WARNING" = "True" ] || [ "$SUPPRESS_FUTURE_WARNING" = "true" ]; then
    export PYTHONWARNINGS="ignore::FutureWarning${PYTHONWARNINGS:+,$PYTHONWARNINGS}"
fi

if [ "$USE_HYBRID_ATTENTION" = "True" ] || [ "$USE_HYBRID_ATTENTION" = "true" ]; then
    USE_HYBRID_ATTENTION=True
    if [ "$ALLOW_FLASH_HYBRID_ATTENTION" != "True" ] && [ "$ALLOW_FLASH_HYBRID_ATTENTION" != "true" ]; then
        ATTN_IMPLEMENTATION=eager
    fi
else
    USE_HYBRID_ATTENTION=False
fi

if [ "$SYNC_TRANSFORMERS_PATCH" = "True" ] || [ "$SYNC_TRANSFORMERS_PATCH" = "true" ]; then
    SYNC_TRANSFORMERS_PATCH=True
else
    SYNC_TRANSFORMERS_PATCH=False
fi

if [ "$SINGLE_GPU_MODE" = "True" ] || [ "$SINGLE_GPU_MODE" = "true" ]; then
    SINGLE_GPU_MODE=True
else
    SINGLE_GPU_MODE=False
fi

if [ "$GRADIENT_CHECKPOINTING" = "True" ] || [ "$GRADIENT_CHECKPOINTING" = "true" ]; then
    GRADIENT_CHECKPOINTING=True
else
    GRADIENT_CHECKPOINTING=False
fi

if [ "$RESUME_TRAINING" = "True" ] || [ "$RESUME_TRAINING" = "true" ]; then
    RESUME_TRAINING=True
else
    RESUME_TRAINING=False
fi

if [ "$AUTO_NEW_OUTPUT_DIR" = "True" ] || [ "$AUTO_NEW_OUTPUT_DIR" = "true" ]; then
    AUTO_NEW_OUTPUT_DIR=True
else
    AUTO_NEW_OUTPUT_DIR=False
fi

export ATTN_IMPLEMENTATION
export LOW_CPU_MEM_USAGE
export CUDA_VISIBLE_DEVICES
export CONDA_ENV_NAME
export OUTPUT_DIR
export USE_HYBRID_ATTENTION
export ALLOW_FLASH_HYBRID_ATTENTION
export SYNC_TRANSFORMERS_PATCH
export BATCH_SIZE
export ACC_STEP
export SAVE_STEPS
export LOGGING_STEPS
export MODEL_MAX_LENGTH
export DATALOADER_NUM_WORKERS
export GRADIENT_CHECKPOINTING
export RESUME_TRAINING
export AUTO_NEW_OUTPUT_DIR
export USE_TORCH_COMPILE
export TORCH_COMPILE_MODE

if [ "$RESUME_TRAINING" = "False" ] && [ -d "$OUTPUT_DIR" ]; then
    has_old_state=False
    if [ -f "$OUTPUT_DIR/config.json" ]; then
        has_old_state=True
    elif find "$OUTPUT_DIR" -maxdepth 1 -type d \( -name "checkpoint-*" -o -name "tmp-checkpoint-*" \) | grep -q .; then
        has_old_state=True
    fi

    if [ "$has_old_state" = "True" ] || [ "$has_old_state" = "true" ]; then
        if [ "$AUTO_NEW_OUTPUT_DIR" = "True" ]; then
            timestamp=$(date +%Y%m%d_%H%M%S)
            OUTPUT_DIR="${OUTPUT_DIR}-fresh-${timestamp}"
            export OUTPUT_DIR
        else
            echo "Error: output directory already contains checkpoints/config and RESUME_TRAINING=False: $OUTPUT_DIR" >&2
            exit 1
        fi
    fi
fi

# Activate environment if needed
if [ -z "${CONDA_PREFIX:-}" ] && command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV_NAME"
fi

if [ "$SYNC_TRANSFORMERS_PATCH" = "True" ] || [ "$SYNC_TRANSFORMERS_PATCH" = "true" ]; then
    site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
    if [ -d "./vila_u/train/transformers_replace/" ]; then
        cp -r ./vila_u/train/transformers_replace/* "$site_pkg_path/transformers/"
    fi
fi

# SLURM configuration
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

echo "MASTER_ADDR=$MASTER_ADDR"
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

echo "GLOBAL_BATCH_SIZE=$global_bs"
echo "PER_DEVICE_TRAIN_BATCH_SIZE=$bs"
echo "GRADIENT_ACCUMULATION_STEPS=$acc_step"

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
echo "  Model Max Length: $MODEL_MAX_LENGTH"
echo "  Dataloader Workers: $DATALOADER_NUM_WORKERS"
echo "  Gradient Checkpointing: $GRADIENT_CHECKPOINTING"
echo "  Action Chunk Size: $ACTION_CHUNK_SIZE"
echo "  Save Steps: $SAVE_STEPS"
echo "  Logging Steps: $LOGGING_STEPS"
echo "  Suppress FutureWarning: $SUPPRESS_FUTURE_WARNING"
echo "  Attention Backend: $ATTN_IMPLEMENTATION"
echo "  Low CPU Mem Usage: $LOW_CPU_MEM_USAGE"
echo "  Use DeepSpeed: $USE_DEEPSPEED"
echo "  Use Hybrid Attention: $USE_HYBRID_ATTENTION"
echo "  Allow Flash Hybrid Attention: $ALLOW_FLASH_HYBRID_ATTENTION"
echo "  Sync Transformers Patch: $SYNC_TRANSFORMERS_PATCH"
echo "  Resume Training: $RESUME_TRAINING"
echo "  Auto New Output Dir: $AUTO_NEW_OUTPUT_DIR"
echo "=========================================="

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
    --logging_steps "$LOGGING_STEPS"
    --tf32 True
    --model_max_length "$MODEL_MAX_LENGTH"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --lazy_preprocess True
    --report_to "$REPORT_TO"
    --use_hybrid_attention "$USE_HYBRID_ATTENTION"
    --action_chunk_size "$ACTION_CHUNK_SIZE"
    --action_dim "$ACTION_DIM"
    --remove_pause_intervals "$REMOVE_PAUSE_INTERVALS"
    --pause_threshold "$PAUSE_THRESHOLD"
)

if [ "$GRADIENT_CHECKPOINTING" = "True" ]; then
    train_args+=(--gradient_checkpointing True)
fi

if [ "$USE_DEEPSPEED" = "True" ] || [ "$USE_DEEPSPEED" = "true" ]; then
    train_args+=(--deepspeed ./scripts/zero2.json)
fi

if [ "$SINGLE_GPU_MODE" = "True" ] || [ "$SINGLE_GPU_MODE" = "true" ]; then
    python -m vila_u.train.train_action_prediction_mem "${train_args[@]}"
else
    torchrun --nnodes=$n_node --nproc_per_node=$NUM_GPUS --master_port=$MASTER_PORT \
        --master_addr $MASTER_ADDR --node_rank=${CURRENT_RANK} \
        vila_u/train/train_action_prediction_mem.py "${train_args[@]}"
fi
