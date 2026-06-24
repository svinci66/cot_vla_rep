"""
VILA-U Action Prediction Training (Main)
Based on VILA-U's train.py framework with action prediction support
"""

import logging
import os
import pathlib
import shutil
import torch
import transformers

from typing import Dict, Optional, Tuple, cast
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from transformers import TrainerCallback

# Import vila_u modules (these don't trigger the numpy issue)
from vila_u import conversation as conversation_lib
from vila_u.model import VILAULlamaModel, VILAULlamaConfig
from vila_u.model.multimodal_encoder.rqvaesigliptransformer_encoder import RQVAESIGLIPTransformerVisionTower
from vila_u.train.vila_u_trainer import VILAUTrainer
from vila_u.train.args import DataArguments, TrainingArguments, ModelArguments
from vila_u.train.callbacks.autoresume_callback import AutoResumeCallback
from vila_u.train.utils import (
    get_checkpoint_path,
    prepare_config_for_training,
    mprint,
)
from vila_u.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
    ACTION_NUM_BINS,
    IGNORE_INDEX,
)
from vila_u.utils.action_tokenizer import (
    actions_to_token_ids,
    build_typed_action_slot_token_ids,
    compute_percentile_action_bin_edges,
    compute_selected_token_logits,
    select_action_token_ids,
    token_ids_to_bins,
)
from vila_u.utils.libero_action import libero_raw_actions_to_model_actions
from vila_u.utils.hybrid_attention import (
    build_action_token_position_mask,
    build_hybrid_attention_mask,
)
from vila_u.utils.tokenizer import tokenize_conversation

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA-U-Action-Prediction"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class ActionPredictionArguments:
    """Arguments for action prediction training"""
    data_root: str = field(
        metadata={"help": "Root directory of LIBERO Goal dataset"}
    )
    action_chunk_size: int = field(
        default=10,
        metadata={"help": "Number of action steps to predict"}
    )
    action_dim: int = field(
        default=7,
        metadata={"help": "Dimension of action space (7-DoF)"}
    )
    image_size: int = field(
        default=256,
        metadata={"help": "Image resolution (will be resized to 256x256)"}
    )
    remove_pause_intervals: bool = field(
        default=True,
        metadata={"help": "Remove pause intervals from trajectories"}
    )
    pause_threshold: float = field(
        default=0.01,
        metadata={"help": "Threshold for detecting pause (L2 norm of action)"}
    )
    max_task_files: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of LIBERO HDF5 task files for small overfit/debug runs"}
    )
    max_demos_per_task: Optional[int] = field(
        default=None,
        metadata={"help": "Limit the number of demonstrations per task file for small overfit/debug runs"}
    )
    task_file: Optional[str] = field(
        default=None,
        metadata={"help": "Exact LIBERO HDF5 task filename to load for debug/overfit runs"}
    )
    task_file_pattern: Optional[str] = field(
        default=None,
        metadata={"help": "Substring or regex used to select LIBERO HDF5 task files"}
    )
    gripper_pause_threshold: float = field(
        default=1e-6,
        metadata={"help": "Maximum adjacent gripper command/state delta considered unchanged for no-op filtering"}
    )
    use_discrete_action_prediction: bool = field(
        default=True,
        metadata={"help": "Use autoregressive discrete action tokens instead of regression"}
    )
    use_hybrid_attention: bool = field(
        default=True,
        metadata={"help": "Use full attention inside the action token block"}
    )
    use_visual_cot: bool = field(
        default=False,
        metadata={"help": "Return a future video frame as subgoal image for Phase 4 Visual CoT"}
    )
    subgoal_min_offset: int = field(
        default=1,
        metadata={"help": "Minimum future-frame offset when sampling Phase 4 subgoal images"}
    )
    subgoal_max_offset: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum future-frame offset when sampling Phase 4 subgoal images; defaults to 10"}
    )
    subgoal_sampling_strategy: str = field(
        default="fixed",
        metadata={"help": "Subgoal frame sampling strategy: uniform or fixed"}
    )
    use_visual_cot_loss: bool = field(
        default=False,
        metadata={"help": "Train subgoal visual token prediction with VILA-U RQTransformer"}
    )
    visual_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for Phase 4 visual subgoal residual-code loss"}
    )
    action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for discrete action token loss"}
    )
    xyz_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Additional CE weight for XYZ translation action tokens"}
    )
    gripper_close_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Additional CE weight for gripper close action tokens in model action space"}
    )
    gripper_transition_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Additional CE weight for gripper action tokens at open/close transitions"}
    )
    use_action_percentile_bins: bool = field(
        default=False,
        metadata={"help": "Use per-dimension action bin edges from training-set percentiles"}
    )
    action_bin_low_percentile: float = field(
        default=1.0,
        metadata={"help": "Lower percentile for per-dimension action bin edges"}
    )
    action_bin_high_percentile: float = field(
        default=99.0,
        metadata={"help": "Upper percentile for per-dimension action bin edges"}
    )
    tune_depth_transformer: bool = field(
        default=True,
        metadata={"help": "Train VILA-U RQTransformer/depth transformer while keeping RQVAE/SigLIP frozen"}
    )


class ActionPredictionDataCollator:
    def __init__(self, tokenizer, model_max_length: int, mm_use_im_start_end: bool):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end

    def __call__(self, batch):
        images = torch.stack([item["observations"] for item in batch])
        action_labels = torch.stack([item["action_labels"] for item in batch])

        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        prompts = [
            tokenize_conversation(
                [{"from": "human", "value": f"{image_token}\n{item['instructions']}"}],
                self.tokenizer,
                add_generation_prompt=True,
            )
            for item in batch
        ]
        input_ids = pad_sequence(
            prompts,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "action_labels": action_labels,
        }
        if "subgoal_images" in batch[0]:
            output["subgoal_images"] = torch.stack([item["subgoal_images"] for item in batch])
            output["subgoal_timesteps"] = torch.tensor(
                [item["subgoal_timestep"] for item in batch],
                dtype=torch.long,
            )
        return output


class DiscreteActionPredictionDataCollator:
    def __init__(
        self,
        tokenizer,
        model_max_length: int,
        mm_use_im_start_end: bool,
        action_token_ids,
        action_slot_token_ids,
        action_chunk_size: int,
        action_dim: int,
        use_hybrid_attention: bool,
        action_bin_edges=None,
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.action_token_ids = action_token_ids
        self.action_slot_token_ids = action_slot_token_ids
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim
        self.num_action_tokens = action_chunk_size * action_dim
        self.use_hybrid_attention = use_hybrid_attention
        self.action_bin_edges = action_bin_edges

    def __call__(self, batch):
        images = torch.stack([item["observations"] for item in batch])
        actions = torch.stack([item["action_labels"] for item in batch])

        image_token = DEFAULT_IMAGE_TOKEN
        if self.mm_use_im_start_end:
            image_token = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN

        input_id_list = []
        label_list = []
        for item, action in zip(batch, actions):
            prompt_ids = tokenize_conversation(
                [{"from": "human", "value": f"{image_token}\n{item['instructions']}"}],
                self.tokenizer,
                add_generation_prompt=True,
            )
            action_token_ids = actions_to_token_ids(
                action,
                self.action_token_ids,
                num_bins=ACTION_NUM_BINS,
                bin_edges=self.action_bin_edges,
            ).view(-1)
            if self.use_hybrid_attention:
                action_input_ids = build_typed_action_slot_token_ids(
                    self.action_slot_token_ids,
                    action_chunk_size=self.action_chunk_size,
                    action_dim=self.action_dim,
                    device=action_token_ids.device,
                    dtype=action_token_ids.dtype,
                )
            else:
                action_input_ids = action_token_ids
            input_ids = torch.cat([prompt_ids, action_input_ids], dim=0)
            labels = torch.cat(
                [
                    torch.full_like(prompt_ids, IGNORE_INDEX),
                    action_token_ids,
                ],
                dim=0,
            )
            input_id_list.append(input_ids[: self.model_max_length])
            label_list.append(labels[: self.model_max_length])

        input_ids = pad_sequence(
            input_id_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            label_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        output = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "labels": labels,
            "action_labels": actions,
        }
        if "previous_action_label" in batch[0]:
            output["previous_action_label"] = torch.stack(
                [item["previous_action_label"] for item in batch]
            )
        if "subgoal_images" in batch[0]:
            output["subgoal_images"] = torch.stack([item["subgoal_images"] for item in batch])
            output["subgoal_timesteps"] = torch.tensor(
                [item["subgoal_timestep"] for item in batch],
                dtype=torch.long,
            )
        return output


class ActionPredictionTrainer(VILAUTrainer):
    def _load_rng_state(self, checkpoint):
        """Load Trainer RNG state under PyTorch 2.6+.

        PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
        Hugging Face Trainer RNG checkpoints can contain NumPy RNG objects, so
        the default safe weights-only loader rejects trusted local
        ``rng_state.pth`` files. During this narrow Trainer RNG restore call,
        force the historical behavior.
        """

        original_torch_load = torch.load

        def torch_load_with_rng_objects(*args, **kwargs):
            kwargs.setdefault("weights_only", False)
            return original_torch_load(*args, **kwargs)

        torch.load = torch_load_with_rng_objects
        try:
            return super()._load_rng_state(checkpoint)
        finally:
            torch.load = original_torch_load

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Support VILA-U component-style checkpoints.

        VILA-U saves checkpoints as component directories (``llm/``,
        ``vision_tower/``, ``mm_projector/``) plus trainer state files instead
        of a top-level ``pytorch_model.bin``/``model.safetensors``. The model is
        already constructed from ``config.resume_path`` before Trainer starts,
        so asking the base Hugging Face Trainer to load model weights again
        raises "Can't find a valid checkpoint". For these component-style
        checkpoints we only skip that redundant model-weight load; Trainer still
        restores optimizer/scheduler/scaler/rng/trainer_state in its normal
        resume flow.
        """

        if resume_from_checkpoint:
            checkpoint_dir = pathlib.Path(str(resume_from_checkpoint))
            has_component_weights = any(
                (checkpoint_dir / component).is_dir()
                for component in ("llm", "vision_tower", "mm_projector")
            )
            has_trainer_state = (checkpoint_dir / "trainer_state.json").is_file()
            has_hf_weights = any(
                (checkpoint_dir / filename).is_file()
                for filename in (
                    "pytorch_model.bin",
                    "model.safetensors",
                    "adapter_model.bin",
                    "adapter_model.safetensors",
                )
            )
            if has_component_weights and has_trainer_state and not has_hf_weights:
                mprint(
                    "Detected VILA-U component checkpoint; "
                    "skipping Hugging Face model-weight reload and resuming trainer state."
                )
                return

        return super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        core_model = model
        while hasattr(core_model, "module"):
            core_model = core_model.module

        if getattr(core_model.config, "use_discrete_action_prediction", False):
            if getattr(core_model.config, "use_hybrid_attention", False):
                (
                    _,
                    position_ids,
                    mm_attention_mask,
                    past_key_values,
                    inputs_embeds,
                    mm_labels,
                ) = core_model.prepare_inputs_labels_for_multimodal(
                    input_ids=inputs["input_ids"],
                    position_ids=None,
                    attention_mask=inputs["attention_mask"],
                    past_key_values=None,
                    labels=inputs["labels"],
                    images=inputs["images"],
                )
                action_token_count = (
                    core_model.config.action_chunk_size * core_model.config.action_dim
                )
                subgoal_prediction_mask = None
                if inputs.get("subgoal_images") is not None:
                    subgoal_embeds, subgoal_codes = core_model.encode_images(
                        inputs["subgoal_images"],
                        image_ids=None,
                    )
                    (
                        inputs_embeds,
                        mm_labels,
                        mm_attention_mask,
                        position_ids,
                        subgoal_prediction_mask,
                    ) = insert_subgoal_embeds_before_action_block(
                        inputs_embeds=inputs_embeds,
                        labels=mm_labels,
                        attention_mask=mm_attention_mask,
                        position_ids=position_ids,
                        subgoal_embeds=subgoal_embeds,
                        num_action_tokens=action_token_count,
                    )
                hybrid_attention_mask = build_hybrid_attention_mask(
                    mm_attention_mask,
                    num_action_tokens=action_token_count,
                    dtype=inputs_embeds.dtype,
                )
                outputs = model(
                    input_ids=None,
                    attention_mask=hybrid_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=None,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    repack_multimodal=False,
                    return_llm_outputs=True,
                    seqlens_in_batch=None,
                )
                labels = mm_labels[:, :, 0]
                action_position_mask = build_action_token_position_mask(
                    mm_attention_mask,
                    num_action_tokens=action_token_count,
                )
                batch_size = outputs.last_hidden_state.shape[0]
                action_hidden_states = outputs.last_hidden_state[action_position_mask].view(
                    batch_size,
                    action_token_count,
                    outputs.last_hidden_state.size(-1),
                )
                action_labels = labels[action_position_mask].view(
                    batch_size,
                    action_token_count,
                )
                action_logits = compute_selected_token_logits(
                    action_hidden_states,
                    core_model.llm.lm_head,
                    core_model.config.action_token_ids,
                )
                action_label_bins = token_ids_to_bins(
                    action_labels,
                    core_model.config.action_token_ids,
                )
                action_loss = torch.nn.functional.cross_entropy(
                    action_logits.reshape(-1, action_logits.size(-1)),
                    action_label_bins.reshape(-1),
                    reduction="none",
                )
                action_loss = action_loss.view(batch_size, action_token_count)
                xyz_loss_weight = float(getattr(core_model.config, "xyz_loss_weight", 1.0))
                gripper_close_loss_weight = float(
                    getattr(core_model.config, "gripper_close_loss_weight", 1.0)
                )
                gripper_transition_loss_weight = float(
                    getattr(core_model.config, "gripper_transition_loss_weight", 1.0)
                )
                action_dim = int(core_model.config.action_dim)
                action_chunk_size = int(core_model.config.action_chunk_size)
                token_weights = torch.ones_like(action_loss)
                if xyz_loss_weight != 1.0:
                    per_token_xyz_mask = torch.zeros(
                        (action_chunk_size, action_dim),
                        device=action_loss.device,
                        dtype=torch.bool,
                    )
                    per_token_xyz_mask[:, : min(3, action_dim)] = True
                    xyz_mask = per_token_xyz_mask.reshape(-1).unsqueeze(0)
                    xyz_weights = torch.full_like(token_weights, xyz_loss_weight)
                    token_weights = torch.where(xyz_mask, xyz_weights, token_weights)
                if (
                    inputs.get("action_labels") is not None
                    and action_dim > 6
                    and (
                        gripper_close_loss_weight != 1.0
                        or gripper_transition_loss_weight != 1.0
                    )
                ):
                    continuous_actions = inputs["action_labels"].to(
                        device=action_loss.device,
                        dtype=torch.float32,
                    )
                    continuous_actions = continuous_actions[:, :action_chunk_size, :action_dim]
                    per_token_gripper_mask = torch.zeros(
                        (action_chunk_size, action_dim),
                        device=action_loss.device,
                        dtype=torch.bool,
                    )
                    per_token_gripper_mask[:, 6] = True
                    gripper_mask = per_token_gripper_mask.reshape(-1).unsqueeze(0)
                    gripper_values = continuous_actions[..., 6]
                    close_mask = gripper_values < 0
                    transition_mask = torch.zeros_like(close_mask)
                    if inputs.get("previous_action_label") is not None:
                        previous_gripper_values = inputs["previous_action_label"].to(
                            device=action_loss.device,
                            dtype=torch.float32,
                        )[..., 6]
                        transition_mask[:, 0] = (
                            torch.abs(gripper_values[:, 0] - previous_gripper_values) > 1e-6
                        )
                    if action_chunk_size > 1:
                        transition_mask[:, 1:] = (
                            torch.abs(gripper_values[:, 1:] - gripper_values[:, :-1]) > 1e-6
                        )
                    per_step_weights = torch.ones_like(gripper_values)
                    if gripper_close_loss_weight != 1.0:
                        close_weights = torch.full_like(
                            per_step_weights,
                            gripper_close_loss_weight,
                        )
                        per_step_weights = torch.where(
                            close_mask,
                            torch.maximum(per_step_weights, close_weights),
                            per_step_weights,
                        )
                    if gripper_transition_loss_weight != 1.0:
                        transition_weights = torch.full_like(
                            per_step_weights,
                            gripper_transition_loss_weight,
                        )
                        per_step_weights = torch.where(
                            transition_mask,
                            torch.maximum(per_step_weights, transition_weights),
                            per_step_weights,
                        )
                    gripper_weights = torch.ones(
                        (batch_size, action_chunk_size, action_dim),
                        device=action_loss.device,
                        dtype=action_loss.dtype,
                    )
                    gripper_weights[..., 6] = per_step_weights.to(action_loss.dtype)
                    token_weights = torch.where(
                        gripper_mask,
                        gripper_weights.reshape(batch_size, action_token_count),
                        token_weights,
                    )
                    action_loss = (action_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
                else:
                    action_loss = (action_loss * token_weights).sum() / token_weights.sum().clamp_min(1.0)
                loss = action_loss * float(getattr(core_model.config, "action_loss_weight", 1.0))
                visual_loss = None
                if (
                    getattr(core_model.config, "use_visual_cot_loss", False)
                    and subgoal_prediction_mask is not None
                    and inputs.get("subgoal_images") is not None
                ):
                    subgoal_hidden_states = outputs.last_hidden_state[subgoal_prediction_mask].view(
                        batch_size,
                        -1,
                        outputs.last_hidden_state.size(-1),
                    )
                    visual_loss = compute_visual_cot_loss(
                        core_model,
                        subgoal_hidden_states,
                        inputs["subgoal_images"],
                        subgoal_codes=subgoal_codes,
                        subgoal_code_offset=core_model.llm.vocab_size,
                    )
                    loss = loss + visual_loss * float(getattr(core_model.config, "visual_loss_weight", 1.0))
                if return_outputs:
                    output = {"logits": action_logits, "action_loss": action_loss}
                    if visual_loss is not None:
                        output["visual_loss"] = visual_loss
                    return loss, output
                return loss

            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                images=inputs["images"],
                labels=inputs["labels"],
                return_dict=True,
            )
            if return_outputs:
                return outputs.loss, outputs
            return outputs.loss

        core_model.freezed_module_patch()

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        images = inputs["images"]
        action_labels = inputs["action_labels"]

        dummy_labels = torch.full_like(input_ids, IGNORE_INDEX)
        (
            _,
            position_ids,
            mm_attention_mask,
            past_key_values,
            inputs_embeds,
            _,
        ) = core_model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=None,
            labels=dummy_labels,
            images=images,
        )

        outputs = model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=None,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            repack_multimodal=False,
            return_llm_outputs=True,
            seqlens_in_batch=mm_attention_mask.sum(dim=-1, dtype=torch.int32),
        )

        action_pred = core_model.predict_actions(
            outputs.last_hidden_state,
            attention_mask=mm_attention_mask,
        )
        loss = torch.nn.functional.l1_loss(action_pred.float(), action_labels.float())

        if return_outputs:
            return loss, {"action_pred": action_pred}
        return loss


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir, _internal_call=True)
        return

    if trainer.args.should_save:
        state_dict = trainer.model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer.model.save_pretrained(
            output_dir,
            state_dict=cpu_state_dict,
            save_only_trainable=getattr(trainer.args, "save_only_trainable", False),
        )


class LightweightEvalCheckpointCallback(TrainerCallback):
    """Save model-only checkpoints at selected epochs for eval/debug.

    These checkpoints intentionally do not include optimizer/scheduler/RNG state,
    so they are compact and suitable for offline/online evaluation but not for
    exact training resume.
    """

    def __init__(
        self,
        every_n_epochs: int = 0,
        epoch_list: str = "",
        save_only_trainable: bool = True,
    ):
        self.every_n_epochs = int(every_n_epochs)
        self.epoch_list = self._parse_epoch_list(epoch_list)
        self.save_only_trainable = bool(save_only_trainable)
        self._last_saved_epoch = 0

    @staticmethod
    def _parse_epoch_list(epoch_list: str) -> set[int]:
        epochs = set()
        for value in str(epoch_list or "").split(","):
            value = value.strip()
            if not value:
                continue
            epoch = int(value)
            if epoch <= 0:
                raise ValueError("Lightweight checkpoint epoch list must contain positive integers")
            epochs.add(epoch)
        return epochs

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if self.every_n_epochs <= 0 and not self.epoch_list:
            return control
        if model is None:
            return control
        if not getattr(state, "is_world_process_zero", True):
            return control

        completed_epoch = int(state.epoch or 0)
        if completed_epoch <= 0:
            return control
        if abs(float(state.epoch or 0.0) - completed_epoch) > 1e-6:
            return control
        if self.epoch_list:
            should_save = completed_epoch in self.epoch_list
        else:
            should_save = completed_epoch % self.every_n_epochs == 0
        if not should_save:
            return control
        if completed_epoch == self._last_saved_epoch:
            return control

        output_dir = pathlib.Path(args.output_dir) / f"eval-checkpoint-epoch-{completed_epoch}"
        tmp_output_dir = output_dir.with_name(f"tmp-{output_dir.name}")
        if tmp_output_dir.exists():
            shutil.rmtree(tmp_output_dir)
        tmp_output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = model.state_dict()
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        model.save_pretrained(
            str(tmp_output_dir),
            state_dict=cpu_state_dict,
            save_only_trainable=self.save_only_trainable,
        )
        state.save_to_json(str(tmp_output_dir / "trainer_state.json"))
        (tmp_output_dir / "README.md").write_text(
            "Lightweight evaluation checkpoint.\n\n"
            "- Includes model/config/tokenizer files needed for eval.\n"
            "- Includes trainer_state.json for training-state inspection.\n"
            "- Omits optimizer.pt, scheduler.pt, rng_state.pth, and scaler.pt.\n"
            "- Not intended for exact training resume.\n",
            encoding="utf-8",
        )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        tmp_output_dir.rename(output_dir)
        self._last_saved_epoch = completed_epoch
        print(f"Saved lightweight eval checkpoint to {output_dir}")
        return control


class RankParameterConsistencyCallback(TrainerCallback):
    """Check that trainable parameters stay synchronized across DDP ranks."""

    def __init__(self, atol: float = 1e-4):
        self.atol = float(atol)
        self._last_checked_epoch = 0

    @staticmethod
    def _trainable_parameter_checksum(model) -> torch.Tensor:
        core_model = model
        while hasattr(core_model, "module"):
            core_model = core_model.module

        device = next(core_model.parameters()).device
        checksum = torch.zeros(2, device=device, dtype=torch.float64)
        for parameter in core_model.parameters():
            if not parameter.requires_grad:
                continue
            values = parameter.detach().float()
            checksum[0] += values.sum(dtype=torch.float64)
            checksum[1] += values.square().sum(dtype=torch.float64)
        return checksum

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if model is None:
            return control
        if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
            return control

        completed_epoch = int(state.epoch or 0)
        if completed_epoch <= 0:
            return control
        if abs(float(state.epoch or 0.0) - completed_epoch) > 1e-6:
            return control
        if completed_epoch == self._last_checked_epoch:
            return control

        checksum = self._trainable_parameter_checksum(model)
        gathered = [torch.zeros_like(checksum) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(gathered, checksum)
        stacked = torch.stack(gathered)
        max_abs_diff = (stacked - stacked[0]).abs().max()
        if max_abs_diff.item() > self.atol:
            raise RuntimeError(
                "DDP trainable parameter checksum mismatch across ranks: "
                f"max_abs_diff={max_abs_diff.item():.6g}, checksums={stacked.detach().cpu().tolist()}"
            )
        if getattr(state, "is_world_process_zero", True):
            print(
                "[rank_parameter_check] "
                f"epoch={completed_epoch} world_size={len(gathered)} "
                f"max_abs_diff={max_abs_diff.item():.6g} checksum={stacked[0].detach().cpu().tolist()}",
                flush=True,
            )
        self._last_checked_epoch = completed_epoch
        return control


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer,
    model,
):
    """Resize tokenizer and embedding."""
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _select_reusable_non_action_token_id(tokenizer, excluded_token_ids) -> int:
    """Reuse an existing non-action tokenizer token as a fallback action slot."""
    excluded_token_ids = set(excluded_token_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    added_token_ids = set(added_vocab.values())
    candidate_ids = sorted(set(tokenizer.get_vocab().values()), reverse=True)
    for token_id in candidate_ids:
        if token_id in excluded_token_ids:
            continue
        if token_id in special_ids or token_id in added_token_ids:
            continue
        return int(token_id)
    raise ValueError("Failed to select a reusable tokenizer token for action slots")


def select_action_slot_token_ids(tokenizer, action_token_ids, model=None) -> dict[str, int]:
    """Create typed [x], [theta], and [gripper] slots for parallel action decoding."""

    base_vocab_size = getattr(tokenizer, "vocab_size", None)
    action_token_set = set(action_token_ids)
    if base_vocab_size is not None:
        base_vocab_size = int(base_vocab_size)
        if min(action_token_set) < 3:
            raise ValueError("Need at least three reusable base-vocab tokens before action bins")
        slot_token_ids = {
            "x": min(action_token_set) - 3,
            "theta": min(action_token_set) - 2,
            "gripper": min(action_token_set) - 1,
        }
        unavailable = [
            token_id
            for token_id in slot_token_ids.values()
            if token_id < 0 or token_id >= base_vocab_size or token_id in action_token_set
        ]
        if unavailable:
            raise ValueError(f"Invalid typed action slot token ids: {unavailable}")
        return slot_token_ids

    excluded = set(action_token_ids)
    slot_token_ids = {}
    for key in ("x", "theta", "gripper"):
        token_id = _select_reusable_non_action_token_id(tokenizer, excluded)
        slot_token_ids[key] = token_id
        excluded.add(token_id)
    return slot_token_ids


def select_action_slot_token_id(tokenizer, action_token_ids) -> int:
    return _select_reusable_non_action_token_id(tokenizer, action_token_ids)


def insert_subgoal_embeds_before_action_block(
    inputs_embeds: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor,
    position_ids: torch.Tensor | None,
    subgoal_embeds: torch.Tensor,
    num_action_tokens: int,
    ignore_index: int = IGNORE_INDEX,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Insert GT subgoal image embeddings immediately before action slots.

    Returns a prediction-position mask for visual loss. The mask covers the
    hidden positions that should predict subgoal codes: the token before the
    first subgoal embedding and the first N-1 subgoal embedding positions.
    """
    batch_size, _, hidden_size = inputs_embeds.shape
    subgoal_len = subgoal_embeds.shape[1]
    label_depth = labels.shape[-1]

    new_embeds = []
    new_labels = []
    for batch_idx in range(batch_size):
        valid_mask = attention_mask[batch_idx].bool()
        cur_valid_embeds = inputs_embeds[batch_idx][valid_mask]
        cur_valid_labels = labels[batch_idx][valid_mask]
        valid_len = cur_valid_embeds.shape[0]
        action_start = valid_len - num_action_tokens
        if action_start < 0:
            raise ValueError(
                f"Cannot insert subgoal before action block: valid_len={valid_len}, "
                f"num_action_tokens={num_action_tokens}"
            )

        cur_embeds = torch.cat(
            [
                cur_valid_embeds[:action_start],
                subgoal_embeds[batch_idx],
                cur_valid_embeds[action_start:],
            ],
            dim=0,
        )
        cur_subgoal_labels = torch.full(
            (subgoal_len, label_depth),
            ignore_index,
            dtype=labels.dtype,
            device=labels.device,
        )
        cur_labels = torch.cat(
            [
                cur_valid_labels[:action_start],
                cur_subgoal_labels,
                cur_valid_labels[action_start:],
            ],
            dim=0,
        )
        new_embeds.append(cur_embeds)
        new_labels.append(cur_labels)

    max_len = max(embed.shape[0] for embed in new_embeds)
    padded_embeds = inputs_embeds.new_zeros((batch_size, max_len, hidden_size))
    padded_labels = torch.full(
        (batch_size, max_len, label_depth),
        ignore_index,
        dtype=labels.dtype,
        device=labels.device,
    )
    padded_attention_mask = torch.zeros(
        (batch_size, max_len),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    subgoal_prediction_mask = torch.zeros(
        (batch_size, max_len),
        dtype=torch.bool,
        device=attention_mask.device,
    )
    padded_position_ids = None
    if position_ids is not None:
        padded_position_ids = torch.zeros(
            (batch_size, max_len),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )

    for batch_idx, (cur_embeds, cur_labels) in enumerate(zip(new_embeds, new_labels)):
        cur_len = cur_embeds.shape[0]
        padded_embeds[batch_idx, :cur_len] = cur_embeds
        padded_labels[batch_idx, :cur_len] = cur_labels
        padded_attention_mask[batch_idx, :cur_len] = True
        action_start = cur_len - num_action_tokens
        subgoal_start = action_start - subgoal_len
        if subgoal_start <= 0:
            raise ValueError("Subgoal block needs at least one context token for visual prediction")
        subgoal_prediction_mask[batch_idx, subgoal_start - 1:action_start - 1] = True
        if padded_position_ids is not None:
            padded_position_ids[batch_idx, :cur_len] = torch.arange(
                0,
                cur_len,
                dtype=padded_position_ids.dtype,
                device=padded_position_ids.device,
            )

    return (
        padded_embeds,
        padded_labels,
        padded_attention_mask,
        padded_position_ids,
        subgoal_prediction_mask,
    )


def compute_visual_cot_loss(
    core_model,
    subgoal_hidden_states: torch.Tensor,
    subgoal_images: torch.Tensor,
    subgoal_codes: torch.Tensor | None = None,
    subgoal_code_offset: int = 0,
) -> torch.Tensor:
    vision_tower = core_model.get_vision_tower()
    vision_model = vision_tower.vision_tower
    rqvae = vision_model.rqvaesiglip
    rqtransformer = vision_model.rqtransformer

    vision_param = next(vision_tower.parameters())
    if subgoal_codes is None:
        subgoal_images = subgoal_images.to(
            device=vision_param.device,
            dtype=vision_param.dtype,
            non_blocking=True,
        )
        with torch.no_grad():
            subgoal_codes, _ = rqvae.encode_image(subgoal_images)
    if subgoal_code_offset:
        subgoal_codes = subgoal_codes - int(subgoal_code_offset)
    subgoal_codes = subgoal_codes.reshape(subgoal_codes.shape[0], -1, subgoal_codes.shape[-1]).long()

    visual_logits = rqtransformer(
        subgoal_hidden_states.to(device=vision_param.device, dtype=vision_param.dtype),
        subgoal_codes,
        rqvae,
    )
    batch_size, seq_len, depth, vocab_size = visual_logits.shape
    return torch.nn.functional.cross_entropy(
        visual_logits.reshape(batch_size * seq_len * depth, vocab_size),
        subgoal_codes.reshape(batch_size * seq_len * depth).to(visual_logits.device),
    )


def compute_dataset_action_bin_edges(train_dataset, data_args: ActionPredictionArguments):
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        backend = torch.distributed.get_backend()
        device = torch.device("cuda", torch.cuda.current_device()) if backend == "nccl" else torch.device("cpu")
        if rank == 0:
            action_bin_edges = _compute_dataset_action_bin_edges_local(train_dataset, data_args).to(device)
        else:
            action_bin_edges = torch.empty(
                data_args.action_dim,
                ACTION_NUM_BINS + 1,
                dtype=torch.float32,
                device=device,
            )
        torch.distributed.broadcast(action_bin_edges, src=0)
        return action_bin_edges.cpu()

    return _compute_dataset_action_bin_edges_local(train_dataset, data_args)


def _compute_dataset_action_bin_edges_local(train_dataset, data_args: ActionPredictionArguments):
    action_chunks = []
    for sample in train_dataset.samples:
        with torch.no_grad():
            import h5py
            with h5py.File(sample['file'], 'r') as h5_file:
                demo = h5_file['data'][sample['demo']]
                if data_args.remove_pause_intervals:
                    non_pause_indices = sample['non_pause_indices']
                    filtered_t = sample['filtered_timestep']
                    action_indices = non_pause_indices[
                        filtered_t : filtered_t + data_args.action_chunk_size
                    ]
                    actions = demo['actions'][action_indices]
                else:
                    timestep = sample['timestep']
                    actions = demo['actions'][timestep : timestep + data_args.action_chunk_size]
        actions = libero_raw_actions_to_model_actions(actions)
        action_chunks.append(torch.as_tensor(actions, dtype=torch.float32))

    if not action_chunks:
        raise ValueError("Cannot compute action bin edges from an empty dataset")

    all_actions = torch.cat(action_chunks, dim=0).clamp(-1.0, 1.0)
    return compute_percentile_action_bin_edges(
        all_actions,
        num_bins=ACTION_NUM_BINS,
        low_percentile=data_args.action_bin_low_percentile,
        high_percentile=data_args.action_bin_high_percentile,
    )


def make_action_prediction_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: ActionPredictionArguments,
    image_processor,
    training_args: TrainingArguments,
    mm_use_im_start_end: bool,
    action_token_ids,
    action_slot_token_ids,
    action_bin_edges=None,
) -> Dict:
    """Create data module for action prediction training"""
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    train_dataset = LiberoGoalDataset(
        data_root=data_args.data_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        action_chunk_size=data_args.action_chunk_size,
        image_size=data_args.image_size,
        remove_pause_intervals=data_args.remove_pause_intervals,
        pause_threshold=data_args.pause_threshold,
        include_subgoal_image=data_args.use_visual_cot,
        subgoal_min_offset=data_args.subgoal_min_offset,
        subgoal_max_offset=data_args.subgoal_max_offset,
        subgoal_sampling_strategy=data_args.subgoal_sampling_strategy,
        max_task_files=data_args.max_task_files,
        max_demos_per_task=data_args.max_demos_per_task,
        task_file=data_args.task_file,
        task_file_pattern=data_args.task_file_pattern,
        gripper_pause_threshold=data_args.gripper_pause_threshold,
    )
    training_args.sample_lens = [len(train_dataset)]

    if data_args.use_discrete_action_prediction and data_args.use_action_percentile_bins and action_bin_edges is None:
        action_bin_edges = compute_dataset_action_bin_edges(train_dataset, data_args)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        action_bin_edges=action_bin_edges,
        data_collator=(
            DiscreteActionPredictionDataCollator(
                tokenizer=tokenizer,
                model_max_length=training_args.model_max_length,
                mm_use_im_start_end=mm_use_im_start_end,
                action_token_ids=action_token_ids,
                action_slot_token_ids=action_slot_token_ids,
                action_chunk_size=data_args.action_chunk_size,
                action_dim=data_args.action_dim,
                use_hybrid_attention=data_args.use_hybrid_attention,
                action_bin_edges=action_bin_edges,
            )
            if data_args.use_discrete_action_prediction
            else ActionPredictionDataCollator(
                tokenizer=tokenizer,
                model_max_length=training_args.model_max_length,
                mm_use_im_start_end=mm_use_im_start_end,
            )
        ),
    )


def train():
    global local_rank

    from transformers import HfArgumentParser, AutoConfig, set_seed

    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        ActionPredictionArguments
    ))
    model_args, data_args, training_args, action_args = cast(
        Tuple[ModelArguments, DataArguments, TrainingArguments, ActionPredictionArguments],
        parser.parse_args_into_dataclasses()
    )

    training_args.run_name = training_args.output_dir.split("/")[-1]
    local_rank = training_args.local_rank
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    set_seed(training_args.seed)

    resume_training = env_flag("RESUME_TRAINING", True)
    resume_path, continue_training = get_checkpoint_path(training_args.output_dir)
    if not resume_training:
        resume_path = None
        continue_training = True

    if not continue_training:
        print(f"Models has been ready under {training_args.output_dir}. Skip training")
        exit(0)

    if resume_path:
        resume_from_checkpoint = True
        config = AutoConfig.from_pretrained(resume_path, trust_remote_code=True)
        config.resume_path = resume_path
        model_cls = eval(config.architectures[0])
    else:
        resume_from_checkpoint = False
        model_cls = VILAULlamaModel
        config = VILAULlamaConfig.from_pretrained(
            model_args.model_name_or_path,
            resume=resume_from_checkpoint
        )
        if getattr(config, "resume_path", None) is not None:
            config.resume_path = model_args.model_name_or_path

    prepare_config_for_training(config, model_args, training_args, data_args)

    if action_args.use_visual_cot and not action_args.use_discrete_action_prediction:
        raise ValueError("Phase 4 Visual CoT currently requires discrete action prediction")
    if action_args.use_visual_cot and not action_args.use_hybrid_attention:
        raise ValueError("Phase 4 Visual CoT requires hybrid attention for action slots")
    if action_args.use_visual_cot_loss and not action_args.use_visual_cot:
        raise ValueError("Visual CoT loss requires use_visual_cot=True")

    # Enable action prediction
    config.use_discrete_action_prediction = action_args.use_discrete_action_prediction
    config.use_action_prediction = not action_args.use_discrete_action_prediction
    config.use_hybrid_attention = action_args.use_hybrid_attention
    config.use_visual_cot = action_args.use_visual_cot
    config.subgoal_min_offset = action_args.subgoal_min_offset
    config.subgoal_max_offset = action_args.subgoal_max_offset
    config.subgoal_sampling_strategy = action_args.subgoal_sampling_strategy
    config.use_visual_cot_loss = action_args.use_visual_cot_loss
    config.visual_loss_weight = action_args.visual_loss_weight
    config.action_loss_weight = action_args.action_loss_weight
    config.xyz_loss_weight = action_args.xyz_loss_weight
    config.gripper_close_loss_weight = action_args.gripper_close_loss_weight
    config.gripper_transition_loss_weight = action_args.gripper_transition_loss_weight
    config.action_dim = action_args.action_dim
    config.action_chunk_size = action_args.action_chunk_size
    config.action_num_bins = ACTION_NUM_BINS
    config.use_action_percentile_bins = action_args.use_action_percentile_bins
    config.action_bin_low_percentile = action_args.action_bin_low_percentile
    config.action_bin_high_percentile = action_args.action_bin_high_percentile
    config.tune_depth_transformer = action_args.tune_depth_transformer
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    if action_args.use_hybrid_attention and attn_implementation == "flash_attention_2":
        attn_implementation = "eager"
    low_cpu_mem_usage = env_flag("LOW_CPU_MEM_USAGE", True)

    model = model_cls(
        config=config,
        attn_implementation=attn_implementation,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    mprint(model)

    model.llm.config.use_cache = False
    model.get_llm().requires_grad_(training_args.tune_language_model)
    mprint(f"Tunable parameters:\nlanguage model {training_args.tune_language_model}")

    if model.get_vision_tower():
        model.get_vision_tower().requires_grad_(training_args.tune_vision_tower)
        model.get_mm_projector().requires_grad_(training_args.tune_mm_projector)
        if isinstance(model.get_vision_tower(), RQVAESIGLIPTransformerVisionTower):
            model.get_vision_tower().vision_tower.rqvaesiglip.requires_grad_(False)
            model.get_vision_tower().vision_tower.rqvaesiglip.eval()
            model.get_vision_tower().vision_tower.rqtransformer.requires_grad_(
                action_args.tune_depth_transformer
            )
            if action_args.tune_depth_transformer:
                model.get_vision_tower().vision_tower.rqtransformer.train()
            else:
                model.get_vision_tower().vision_tower.rqtransformer.eval()
        else:
            raise NotImplementedError()
        print(f"vision tower {training_args.tune_vision_tower}")
        print(f"depth transformer {action_args.tune_depth_transformer}")
        print(f"mm projector {training_args.tune_mm_projector}")

    # Action head is always trainable
    if hasattr(model, 'action_head'):
        model.action_head.requires_grad_(True)
        print(f"action head True")

    if not any([
        training_args.tune_language_model,
        training_args.tune_vision_tower,
        action_args.tune_depth_transformer,
        training_args.tune_mm_projector,
        hasattr(model, 'action_head')
    ]):
        logging.warning(
            "You are not tuning any part of the model. Please check if this is intended."
        )

    def need_to_modify_do_sample(generation_config):
        if generation_config.do_sample is False:
            if (
                generation_config.temperature is not None
                and generation_config.temperature != 1.0
            ):
                return True
            if generation_config.top_p is not None and generation_config.top_p != 1.0:
                return True
        return False

    if need_to_modify_do_sample(model.llm.generation_config):
        model.llm.generation_config.do_sample = True

    if training_args.gradient_checkpointing:
        if hasattr(model.llm, "enable_input_require_grads"):
            model.llm.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = model.tokenizer
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model.llm,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    model.llm.pad_token_id = tokenizer.pad_token_id
    model.llm.config.tokenizer_padding_side = tokenizer.padding_side
    model.llm.config.tokenizer_model_max_length = tokenizer.model_max_length

    vision_tower = model.get_vision_tower()
    if vision_tower is None:
        raise ValueError("Action prediction training requires a vision tower")

    data_args.image_processor = vision_tower.image_processor
    data_args.is_multimodal = True
    model.config.num_video_frames = data_args.num_video_frames
    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
        model_args.mm_use_im_start_end
    )
    model.config.mm_use_vi_start_end = data_args.mm_use_vi_start_end = (
        model_args.mm_use_vi_start_end
    )
    model.config.mm_projector_lr = training_args.mm_projector_lr
    training_args.use_im_start_end = model_args.mm_use_im_start_end
    training_args.use_vi_start_end = model_args.mm_use_vi_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)
    action_token_ids = None
    action_slot_token_ids = None
    if action_args.use_discrete_action_prediction:
        action_token_ids = select_action_token_ids(tokenizer, num_bins=ACTION_NUM_BINS)
        action_slot_token_ids = select_action_slot_token_ids(tokenizer, action_token_ids, model=model)
        model.config.action_token_ids = action_token_ids
        model.config.action_num_bins = ACTION_NUM_BINS
        model.config.action_slot_token_ids = action_slot_token_ids
        model.config.action_slot_token_id = action_slot_token_ids["x"]

    # Create data module for action prediction
    data_module = make_action_prediction_data_module(
        tokenizer=tokenizer,
        data_args=action_args,
        image_processor=vision_tower.image_processor,
        training_args=training_args,
        mm_use_im_start_end=data_args.mm_use_im_start_end,
        action_token_ids=action_token_ids,
        action_slot_token_ids=action_slot_token_ids,
    )

    action_bin_edges = data_module.pop("action_bin_edges", None)
    if action_bin_edges is not None:
        model.config.action_bin_edges = action_bin_edges.cpu().tolist()
        model.config.action_bin_low_percentile = action_args.action_bin_low_percentile
        model.config.action_bin_high_percentile = action_args.action_bin_high_percentile
        model.config.use_action_percentile_bins = action_args.use_action_percentile_bins
        print(
            "action percentile bins "
            f"{action_args.action_bin_low_percentile}-{action_args.action_bin_high_percentile}: "
            f"shape={tuple(action_bin_edges.shape)}"
        )

    # Custom trainer for action prediction
    trainer = ActionPredictionTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Add auto-resume callback
    trainer.add_callback(AutoResumeCallback())
    lightweight_checkpoint_epochs = int(getattr(training_args, "lightweight_eval_checkpoint_epochs", 0))
    lightweight_checkpoint_epoch_list = getattr(training_args, "lightweight_eval_checkpoint_epoch_list", "")
    if lightweight_checkpoint_epochs > 0 or lightweight_checkpoint_epoch_list:
        trainer.add_callback(
            LightweightEvalCheckpointCallback(
                every_n_epochs=lightweight_checkpoint_epochs,
                epoch_list=lightweight_checkpoint_epoch_list,
                save_only_trainable=getattr(training_args, "save_only_trainable", True),
            )
        )
    if bool(getattr(training_args, "rank_parameter_check", False)):
        trainer.add_callback(RankParameterConsistencyCallback())

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
