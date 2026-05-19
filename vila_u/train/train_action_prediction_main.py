"""
VILA-U Action Prediction Training (Main)
Based on VILA-U's train.py framework with action prediction support
"""

import logging
import os
import pathlib
import torch
import transformers

from typing import Dict, Optional, Tuple, cast
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence

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
    compute_percentile_action_bin_edges,
    compute_selected_token_logits,
    select_action_token_ids,
    token_ids_to_bins,
)
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
        default=None,
        metadata={"help": "Maximum future-frame offset when sampling Phase 4 subgoal images; defaults to action_chunk_size"}
    )
    subgoal_sampling_strategy: str = field(
        default="uniform",
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
    use_action_percentile_bins: bool = field(
        default=True,
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
        action_slot_token_id: int,
        action_chunk_size: int,
        action_dim: int,
        use_hybrid_attention: bool,
        action_bin_edges=None,
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.action_token_ids = action_token_ids
        self.action_slot_token_id = action_slot_token_id
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
                action_input_ids = torch.full_like(
                    action_token_ids,
                    fill_value=self.action_slot_token_id,
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
        }
        if "subgoal_images" in batch[0]:
            output["subgoal_images"] = torch.stack([item["subgoal_images"] for item in batch])
            output["subgoal_timesteps"] = torch.tensor(
                [item["subgoal_timestep"] for item in batch],
                dtype=torch.long,
            )
        return output


class ActionPredictionTrainer(VILAUTrainer):
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
                outputs = core_model.llm.model(
                    input_ids=None,
                    attention_mask=hybrid_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
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
                )
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

        outputs = core_model.llm.model(
            input_ids=None,
            attention_mask=mm_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
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


def select_action_slot_token_id(tokenizer, action_token_ids) -> int:
    """Reuse an existing non-action tokenizer token as the parallel action slot."""
    action_token_set = set(action_token_ids)
    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    added_token_ids = set(added_vocab.values())
    candidate_ids = sorted(set(tokenizer.get_vocab().values()), reverse=True)
    for token_id in candidate_ids:
        if token_id in action_token_set:
            continue
        if token_id in special_ids or token_id in added_token_ids:
            continue
        return int(token_id)
    raise ValueError("Failed to select a reusable tokenizer token for action slots")


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
    action_slot_token_id: int | None,
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
                action_slot_token_id=action_slot_token_id,
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
    action_slot_token_id = None
    if action_args.use_discrete_action_prediction:
        action_token_ids = select_action_token_ids(tokenizer, num_bins=ACTION_NUM_BINS)
        action_slot_token_id = select_action_slot_token_id(tokenizer, action_token_ids)
        model.config.action_token_ids = action_token_ids
        model.config.action_num_bins = ACTION_NUM_BINS
        model.config.action_slot_token_id = action_slot_token_id

    # Create data module for action prediction
    data_module = make_action_prediction_data_module(
        tokenizer=tokenizer,
        data_args=action_args,
        image_processor=vision_tower.image_processor,
        training_args=training_args,
        mm_use_im_start_end=data_args.mm_use_im_start_end,
        action_token_ids=action_token_ids,
        action_slot_token_id=action_slot_token_id,
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

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_state()

    model.llm.config.use_cache = True
    model.config.resume_path = model.config._name_or_path = training_args.output_dir

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
