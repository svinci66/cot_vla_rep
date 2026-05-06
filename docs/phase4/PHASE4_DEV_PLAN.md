# Phase 4 Visual CoT-VLA Development Plan

## Goal

Implement Phase 4 on top of `phase3-hybrid-attention-stable` as a paper-aligned Visual CoT-VLA pipeline:

1. Generate or condition on a future subgoal image as visual chain-of-thought.
2. Predict a chunk of discrete action tokens conditioned on the observation, instruction, and subgoal.
3. Preserve Phase 3 hybrid attention: causal attention for text/image generation and full attention inside the action-token block.
4. Preserve Phase 3 action vocabulary repurposing: reuse 256 existing tokenizer tokens as action bins without expanding the vocabulary for actions.

## Paper Requirements To Preserve

### Hybrid Attention Is Non-Negotiable

Phase 4 must not call the LLM with the default causal mask when action slots are present.

Required mask behavior:

- Text and image/subgoal generation positions remain causal.
- The final action chunk, `action_chunk_size * action_dim` tokens, uses full attention.
- Every action token can attend to all valid context tokens and all other action tokens in the same chunk.

Current reusable implementation:

- `vila_u/utils/hybrid_attention.py` builds the additive hybrid attention mask.
- `vila_u/train/train_action_prediction_main.py` already uses it in Phase 3 training.
- `vila_u/model/vila_u_arch.py` already uses it in Phase 3 inference.

Phase 4 forward paths must follow the same pattern:

```python
hybrid_attention_mask = build_hybrid_attention_mask(
    attention_mask=full_attention_mask,
    num_action_tokens=config.action_chunk_size * config.action_dim,
    dtype=inputs_embeds.dtype,
)
outputs = model.llm.model(
    input_ids=None,
    inputs_embeds=inputs_embeds,
    attention_mask=hybrid_attention_mask,
    position_ids=position_ids,
    use_cache=False,
    return_dict=True,
    seqlens_in_batch=None,
)
```

### Action Vocabulary Repurposing Must Continue

Phase 4 must keep the Phase 3 action-token scheme:

- Continuous 7-DoF actions are discretized into 256 bins.
- The 256 action bins map onto existing tokenizer token IDs.
- The tokenizer vocabulary is not expanded for action bins.
- `lm_head` logits are gathered only for the selected action-token IDs.

Current reusable implementation:

- `vila_u/utils/action_tokenizer.py` selects existing token IDs and maps actions to/from token IDs.
- `compute_selected_token_logits()` avoids computing/storing unused vocabulary logits for the action loss.

## Current Phase 3 Assets To Reuse

- Discrete action tokenizer and detokenizer: `vila_u/utils/action_tokenizer.py`
- Hybrid attention mask: `vila_u/utils/hybrid_attention.py`
- Phase 3 collator/trainer loss shape: `vila_u/train/train_action_prediction_main.py`
- Inference action block path: `vila_u/model/vila_u_arch.py`
- VILA-U RQ image components:
  - `vision_tower.vision_tower.rqvaesiglip.encode_image()` returns RQ-VAE codes.
  - `vision_tower.vision_tower.rqtransformer.forward()` predicts residual image-code depths from LLM hidden states.
  - `vision_tower.vision_tower.rqtransformer.generate()` autoregressively generates image residual codes.
  - `vision_tower.vision_tower.rqvaesiglip.decode()` decodes generated visual embeddings back to images.

## Important Correction To Existing Phase 4 Notes

Any Phase 4 pseudo-code like this is incomplete:

```python
outputs = self.llm.model(inputs_embeds=full_embeds, **kwargs)
```

It silently falls back to standard causal attention. When action tokens are included, Phase 4 must explicitly pass a hybrid attention mask built over the complete sequence, with the action-token block marked as the last valid tokens.

The later example in `/Users/sauvinci/Downloads/PHASE4_ISSUES_AND_SOLUTIONS.md` that adds `build_hybrid_attention_mask(...)` is the direction to follow; the simplified example near the end that omits `attention_mask=hybrid_mask` must not be used as implementation guidance.

## Development Milestones

### Milestone 0: Restore A Clean Phase 3 Baseline

Deliverables:

- Confirm branch starts from `phase3-hybrid-attention-stable`.
- Run focused Phase 3 tests for action tokenization and hybrid masks.
- Record baseline command lines for Phase 3 training and inference checks.

Validation:

- `python tests/test_phase2_action_tokenizer.py`
- `python tests/test_phase3_hybrid_attention.py`

### Milestone 1: Add Future-Frame Subgoal Sampling

Do not build a separate CoT dataset at this stage. Reuse existing LIBERO video sequences and sample future frames online as subgoal images during training.

Implement optional Phase 4 sampling in the existing LIBERO dataset/collator so it returns:

- Current observation image.
- Language instruction.
- Ground-truth action chunk.
- Ground-truth future subgoal image sampled from the same demonstration video.
- Optional metadata: file path, demo ID, timestep, chosen subgoal timestep.

Rules:

- Subgoal offset range should be configurable.
- Clamp subgoal timestep at the trajectory end.
- Use the same image resolution as the paper and existing VILA-U setup: 256 x 256.
- Do not put RQ-VAE image code IDs into `input_ids`.

Implemented direction:

- Extend `vila_u/data/libero_dataset_v2.py` with optional future-frame subgoal sampling.
- Extend existing collators to pass through `subgoal_images` when `use_visual_cot=True`.

Validation:

- Later: unit test sample indexing and end-of-trajectory clamping.
- Later: unit test returned tensor shapes for observation, subgoal, and action chunk.

### Milestone 2: Implement Oracle-Subgoal Action Baseline

Purpose:

- Verify the Phase 4 data flow before training visual generation.
- Establish an upper-bound-style baseline where the model conditions on ground-truth subgoal embeddings.

Implementation:

- Encode observation through the normal VILA-U multimodal path.
- Encode the GT subgoal image through the vision tower/projector path as embeddings.
- Append action slot tokens after the subgoal context.
- Compute action loss exactly like Phase 3 using selected action-token logits.
- Build hybrid attention over the full sequence before `llm.model(...)`.

Required attention layout:

```text
[prompt + observation visual tokens] [GT subgoal visual embeddings] [70 action slots]
causal region                         causal region                 full-action region
```

Deliverables:

- Implemented through the existing Phase 3 trainer path in `vila_u/train/train_action_prediction_main.py`.
- `scripts/train_phase4_oracle_subgoal.sh` enables future-frame subgoal sampling by default.
- `VILAULlamaModel.predict_action(..., subgoal_image=...)` supports oracle-subgoal inference.
- `scripts/check_phase4_oracle_subgoal.py` checks one HDF5 sample end-to-end.
- `scripts/eval_phase4_oracle_subgoal_offline.py` computes offline action MAE/MSE over HDF5 samples.
- Later: tests proving action positions are still the last `70` valid tokens after subgoal insertion.

Validation:

- Later: loss computes without NaN on a tiny mocked batch.
- Later: hybrid mask test confirms action rows can attend bidirectionally within the action block.
- No tokenizer resize is performed for action bins.

### Milestone 3: Add Visual Generation Loss Using Existing RQTransformer

Purpose:

- Align with the paper's visual-token prediction path without incorrectly using text embeddings or `lm_head` for image codes.

Implementation:

- Use `rqvaesiglip.encode_image(subgoal_image)` to obtain GT residual code IDs shaped like `[B, image_tokens, D]`.
- Use LLM hidden states at subgoal prediction positions as code embeddings `h_j`.
- Use `rqtransformer.forward(embed_from_body=subgoal_hidden, code=gt_codes, model_aux=rqvaesiglip)` to predict residual code logits.
- Compute visual cross-entropy over residual depth `D`.
- Combine losses as `loss = visual_loss_weight * visual_loss + action_loss_weight * action_loss`.

Critical design choice:

- Prefer reusing VILA-U's existing `RQTransformer` as the Depth Transformer instead of creating a second new module. The existing implementation already matches the paper concept: it predicts residual codebook depths autoregressively conditioned on LLM body embeddings.

Deliverables:

- Implemented visual loss helper in `vila_u/train/train_action_prediction_main.py`.
- Config flags for `use_visual_cot_loss`, `visual_loss_weight`, and `action_loss_weight`.
- `scripts/train_phase4_visual_cot.sh` enables subgoal residual-code loss.
- Tests with fake codes/logits or a tiny mocked RQTransformer.

Validation:

- `visual_logits` shape is `[B, image_tokens, D, codebook_size]`.
- `gt_subgoal_codes` shape is `[B, image_tokens, D]`.
- Visual loss ignores no action tokens and action loss ignores all visual positions.

### Milestone 4: Implement Two-Stage Inference

Inference should be staged, not a single naive forward:

1. Given observation and instruction, generate subgoal image codes using causal visual generation.
2. Decode or embed the generated subgoal.
3. Append action slots and predict the full action chunk with hybrid attention.
4. Convert predicted action token IDs back to continuous actions.

Implementation notes:

- For subgoal generation, reuse `rqtransformer.generate()` where possible.
- For action prediction, use the Phase 3 hybrid-attention action block path, extended to include generated subgoal embeddings before action slots.
- Keep an oracle-subgoal inference mode for debugging and ablation.

Deliverables:

- `generate_visual_cot_subgoal(...)` generates subgoal embeddings/codes/image.
- `predict_action_with_generated_subgoal(...)` performs two-stage generated-subgoal action prediction.
- `scripts/check_phase4_oracle_subgoal.py --mode generated` checks generated-subgoal inference.
- `scripts/eval_phase4_oracle_subgoal_offline.py --mode generated` evaluates generated-subgoal action prediction offline.
- `scripts/visualize_phase4_subgoals.py --mode both` saves observation, GT subgoal, generated subgoal, and action summary panels.
- `vila_u/eval/trajectory_generator.py` supports `subgoal_mode="generated"` for later LIBERO rollout evaluation.

Validation:

- Generated action tensor shape is `[action_chunk_size, action_dim]`.
- Generated subgoal image shape is `[3, 256, 256]` or `[B, 3, 256, 256]`.
- Hybrid attention is used for the action stage.

### Milestone 5: Training And Ablation Protocol

Start with conservative training modes:

1. `phase4_oracle_subgoal_action_only`: GT subgoal embeddings + action loss only.
2. `phase4_teacher_forced_visual_action`: GT visual codes for visual loss + GT action slots for action loss.
3. `phase4_generated_subgoal_eval`: generated subgoal at inference + action prediction.

Track metrics:

- Action-token cross entropy.
- Visual residual-code cross entropy.
- Detokenized action L1/L2 for interpretability only.
- LIBERO success rate.
- Subgoal reconstruction/generation snapshots.
- Chunk smoothness/consistency metrics from Phase 3 comparison scripts.

Recommended benchmark order:

1. Tiny mocked test.
2. A few real LIBERO trajectories offline.
3. LIBERO-Spatial first.
4. LIBERO-Goal after Spatial is stable.

## Implementation Guardrails

- Do not expand the tokenizer for action bins.
- Do not feed raw image/RQ code IDs through text token embeddings.
- Do not use `lm_head` to predict visual RQ-VAE codes.
- Do not call `llm.model(...)` without an explicit hybrid mask when action slots are present.
- Do not let action labels include prompt, observation, or subgoal positions.
- Do not let visual loss include action positions.
- Keep Phase 3 scripts and behavior intact; add Phase 4 code paths behind new flags or new entrypoints.

## Initial File-Level Plan

Likely new files:

- `vila_u/train/train_visual_cot.py`
- `scripts/train_phase4_oracle_subgoal.sh`
- `scripts/train_phase4_visual_cot.sh`
- `scripts/eval_phase4_visual_cot.py`
- `tests/test_phase4_cot_dataset.py`
- `tests/test_phase4_visual_loss.py`
- `tests/test_phase4_hybrid_attention.py`

Likely modified files:

- `vila_u/constants.py`: add `<subgoal>` / `<act>` markers only if useful as control tokens, not as action-bin tokens.
- `vila_u/model/configuration_vila_u.py`: add Phase 4 config flags.
- `vila_u/model/vila_u_arch.py`: add reusable visual CoT generation/action helper methods.
- `vila_u/utils/__init__.py`: export new helpers if needed.

## Definition Of Done

Phase 4 is considered implementation-complete only when:

- GT subgoal action-only training runs with Phase 3 hybrid attention.
- Visual residual-code loss uses RQTransformer/Depth Transformer, not `lm_head`.
- Generated-subgoal inference works end-to-end.
- Action-token vocabulary repurposing remains unchanged.
- Tests cover dataset shapes, visual-code loss shape, hybrid attention with inserted subgoal embeddings, and action-token decoding.
  Current lightweight coverage: `tests/test_phase4_visual_cot.py` checks causal 4D masks, subgoal insertion before action slots, and visual CoT loss shape.
- Evaluation can compare Phase 3 vs Phase 4 under the same LIBERO suite, seeds, and episode count.
