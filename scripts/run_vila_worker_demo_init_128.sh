#!/bin/bash

# Start the VILA-U ZMQ model worker for demo-init online evaluation.

MODEL_PATH=${MODEL_PATH:-"./checkpoints/vila-u-action-only-full-8gpu-bs64-fixed-lr-2e-5-30ep/eval-checkpoint-epoch-15"}
BIND=${BIND:-"tcp://127.0.0.1:5555"}
DEVICE=${DEVICE:-cuda}
CAMERA=${CAMERA:-agentview_image}
SUBGOAL_MODE=${SUBGOAL_MODE:-none}
DEBUG_JSONL=${DEBUG_JSONL:-}
REPLAN_EVERY_STEP=${REPLAN_EVERY_STEP:-False}

cmd=(
    python scripts/vila_zmq_model_worker.py
    --model-path "$MODEL_PATH"
    --bind "$BIND"
    --device "$DEVICE"
    --camera "$CAMERA"
    --subgoal-mode "$SUBGOAL_MODE"
)

if [ -n "$DEBUG_JSONL" ]; then
    cmd+=(--debug-jsonl "$DEBUG_JSONL")
fi

if [ "$REPLAN_EVERY_STEP" = "True" ] || [ "$REPLAN_EVERY_STEP" = "true" ]; then
    cmd+=(--replan-every-step)
fi

echo "Starting VILA-U model worker"
echo "  Model: $MODEL_PATH"
echo "  Bind: $BIND"
echo "  Device: $DEVICE"
"${cmd[@]}"
