#!/bin/bash

# Start the LIBERO ZMQ env server from demonstration init states.

SUITE=${SUITE:-libero_goal}
TASK_ID=${TASK_ID:-0}
EPISODES=${EPISODES:-1}
MAX_STEPS=${MAX_STEPS:-300}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-5555}
CAMERA=${CAMERA:-agentview_image}
HEIGHT=${HEIGHT:-128}
WIDTH=${WIDTH:-128}
SEED=${SEED:-0}
DEMO_INIT_HDF5=${DEMO_INIT_HDF5:-"/data/share/1919650160032350208/sj/LIBERO/datasets/libero_goal/open_the_middle_drawer_of_the_cabinet_demo.hdf5"}
DEMO_START=${DEMO_START:-0}
DEMO_COUNT=${DEMO_COUNT:-10}
OUTPUT_JSON=${OUTPUT_JSON:-"outputs/libero_demo_init_128_summary.json"}
OUTPUT_DIR=${OUTPUT_DIR:-}
SAVE_ROLLOUT_VIDEO=${SAVE_ROLLOUT_VIDEO:-True}
VIDEO_DIR=${VIDEO_DIR:-}
VIDEO_FORMAT=${VIDEO_FORMAT:-both}
VIDEO_FPS=${VIDEO_FPS:-20}
SAVE_FAILURES=${SAVE_FAILURES:-True}
MUJOCO_GL=${MUJOCO_GL:-egl}

cmd=(
    python scripts/libero_zmq_env_server.py
    --suite "$SUITE"
    --task-id "$TASK_ID"
    --episodes "$EPISODES"
    --max-steps "$MAX_STEPS"
    --host "$HOST"
    --port "$PORT"
    --camera "$CAMERA"
    --height "$HEIGHT"
    --width "$WIDTH"
    --seed "$SEED"
    --demo-init-hdf5 "$DEMO_INIT_HDF5"
    --demo-start "$DEMO_START"
    --demo-count "$DEMO_COUNT"
    --output-json "$OUTPUT_JSON"
    --video-format "$VIDEO_FORMAT"
    --video-fps "$VIDEO_FPS"
    --mujoco-gl "$MUJOCO_GL"
)

if [ -n "$OUTPUT_DIR" ]; then
    cmd+=(--output-dir "$OUTPUT_DIR")
fi

if [ -n "$VIDEO_DIR" ]; then
    cmd+=(--video-dir "$VIDEO_DIR")
fi

if [ "$SAVE_ROLLOUT_VIDEO" = "True" ] || [ "$SAVE_ROLLOUT_VIDEO" = "true" ]; then
    cmd+=(--save-rollout-video)
fi

if [ "$SAVE_FAILURES" = "True" ] || [ "$SAVE_FAILURES" = "true" ]; then
    cmd+=(--save-failures)
fi

echo "Starting LIBERO demo-init env server"
echo "  Suite/Task: $SUITE/$TASK_ID"
echo "  Demo HDF5: $DEMO_INIT_HDF5"
echo "  Demo range: $DEMO_START..$((DEMO_START + DEMO_COUNT - 1))"
echo "  Resolution: ${WIDTH}x${HEIGHT}"
echo "  Output: $OUTPUT_JSON"
"${cmd[@]}"
