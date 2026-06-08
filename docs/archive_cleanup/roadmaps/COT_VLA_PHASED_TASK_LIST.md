# CoT-VLA Phased Task List

## Purpose

This file records the phased reproduction plan that maps the current VILA-U modifications to the CoT-VLA paper's ablation-style progression.

Guiding rule:
- Change one core variable per phase.
- Keep benchmark, seed, evaluation episodes, and logging format fixed across phases.
- Do not move to the next phase before collecting comparable metrics for the current one.

Recommended first benchmark:
- `LIBERO-Spatial`

## Current Status

Current implementation corresponds to:
- `VILA-U + LIBERO + action head + continuous regression + action chunking`

This is a baseline action-prediction system, not full CoT-VLA.

## Phase 1

Name:
- `Regression + Action Chunking Baseline`

Paper mapping:
- Closest to the paper's `+ action chunking` variant, but still uses continuous regression instead of discrete action tokens.

Goals:
- Verify end-to-end training and evaluation are functional.
- Produce the first reproducible baseline success rate.

Do not change:
- Action representation
- Loss type
- Attention mask design
- Visual CoT logic

Tasks:
- Train the current implementation on `LIBERO-Spatial`.
- Run closed-loop evaluation in LIBERO.
- Record training config, checkpoint path, success rate, average steps, and example trajectories.

Expected output:
- A stable baseline number that becomes the reference for all later phases.

## Phase 2

Name:
- `Vanilla VLA Alignment`

Paper mapping:
- The paper's standard `VLA` variant.

Goals:
- Replace continuous regression with autoregressive discrete action-token prediction.

Required changes:
- Discretize each of the 7 continuous action dimensions into 256 bins.
- Reuse 256 low-frequency tokenizer tokens as action tokens.
- Remove the standalone `action_head`.
- Use the LLM `lm_head` to predict action tokens.
- Replace `L1 loss` with `Cross-Entropy loss`.
- Update inference to autoregressively generate action tokens and detokenize them back to continuous actions.

Expected output:
- A working discrete-token VLA baseline aligned with mainstream embodied VLA methods.

## Phase 3

Name:
- `+ Hybrid Attention`

Paper mapping:
- The paper's `+ hybrid attention` variant.

Goals:
- Improve action coherence by allowing full attention within the action-token block.

Required changes:
- Keep text/image generation under causal attention.
- Change action-token attention from causal to full attention.
- Ensure action tokens in the same chunk can see each other.

Evaluation focus:
- Smoother action sequences
- Better chunk consistency
- Success-rate gain over Phase 2

Expected output:
- A discrete-token VLA model with hybrid attention and improved control performance.

## Phase 4

Name:
- `+ Visual CoT`

Paper mapping:
- The paper's full `+ CoT (ours)` variant.

Goals:
- Introduce the paper's visual chain-of-thought stage before action prediction.

Required changes:
- Add subgoal-related special tokens and config fields.
- Use VILA-U's native image-generation path to autoregressively predict future subgoal image tokens.
- After subgoal generation, predict action tokens conditioned on the generated subgoal and prior context.
- Train with joint visual-generation loss and action loss.
- Add a full closed-loop inference path:
  `observation -> subgoal -> action`

Expected output:
- A full CoT-VLA-style pipeline with visual reasoning and action generation.

## Common Rules Across All Phases

- Use the same benchmark first: `LIBERO-Spatial`
- Fix random seeds
- Fix evaluation episode count
- Save metrics and trajectories for every phase
- Keep one output directory per phase
- Compare each phase only against the immediately previous phase

Suggested checkpoint names:
- `checkpoints/phase1_regression_chunking`
- `checkpoints/phase2_discrete_vla`
- `checkpoints/phase3_hybrid_attention`
- `checkpoints/phase4_visual_cot`

Suggested eval directories:
- `eval_results/phase1_spatial`
- `eval_results/phase2_spatial`
- `eval_results/phase3_spatial`
- `eval_results/phase4_spatial`

## Per-Phase Record Template

For each phase, record:
- Benchmark
- Tasks evaluated
- Seed
- Training dataset path
- Model path
- Output directory
- Batch settings
- Image size
- Hardware mode
- Success rate
- Average steps
- Typical failure cases
- Delta versus previous phase

## Stop Conditions

Do not proceed to the next phase if:
- Training is unstable
- Evaluation is not reproducible
- Success rate collapses and the root cause is unclear
- New modules are added before the current phase baseline is established
