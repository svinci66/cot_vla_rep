# LIBERO Online Evaluation With Separate Conda Environments

Use two processes when LIBERO/robosuite and the VILA-U model live in different conda environments.
Do not import both stacks in one Python process.

## Install lightweight IPC dependencies

In both environments:

```bash
pip install pyzmq msgpack msgpack-numpy
```

## Terminal 1: model environment

Run the model worker in the VILA-U/model environment:

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep
python scripts/vila_zmq_model_worker.py \
  --model-path ./checkpoints/vila-u-action-prediction-phase3-fresh-20260424_100759 \
  --bind tcp://127.0.0.1:5555 \
  --device cuda \
  --subgoal-mode none
```

For Phase 4 generated-subgoal inference, use:

```bash
python scripts/vila_zmq_model_worker.py \
  --model-path <phase4_checkpoint_or_output_dir> \
  --bind tcp://127.0.0.1:5555 \
  --device cuda \
  --subgoal-mode generated \
  --cfg 3.0
```

## Terminal 2: LIBERO environment

Run the environment server in the LIBERO environment:

```bash
cd /data/share/1919650160032350208/sj/cot-vla/cot_vla_rep
export MUJOCO_GL=egl
python scripts/libero_zmq_env_server.py \
  --suite libero_goal \
  --task-id 0 \
  --episodes 10 \
  --max-steps 300 \
  --host 127.0.0.1 \
  --port 5555 \
  --output-json outputs/libero_goal_task0_phase3_online.json
```

## Outputs

The server reports online success rate:

```text
success_rate = successful_episodes / total_episodes
```

This is the metric corresponding to LIBERO task success in the CoT-VLA paper tables.
Offline MAE/token accuracy remain useful diagnostics but are not replacements for online success rate.

## Notes

- `env.step()` always runs in the LIBERO process.
- Model inference always runs in the VILA-U process.
- Observations are sent as NumPy arrays through ZeroMQ/msgpack.
- Actions are clipped to `[-1, 1]` before stepping the environment.
- Start with 1 task and 5-10 episodes before running benchmark-scale evaluation.
