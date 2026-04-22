"""
VILA-U Phase 4: Visual Chain-of-Thought VLA Training
Based on train_action_prediction_main.py with CoT-VLA extensions
"""

import logging
import os
import pathlib
import torch
import transformers

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

from typing import Dict, Tuple, cast
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence

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
    ACTION_NUM_BINS,
    IGNORE_INDEX,
    DEFAULT_ACT_START_TOKEN,
    DEFAULT_ACT_END_TOKEN,
    DEFAULT_SUBGOAL_START_TOKEN,
    DEFAULT_SUBGOAL_END_TOKEN,
)
from vila_u.utils.action_tokenizer import (
    compute_selected_token_logits,
    select_action_token_ids,
    token_ids_to_bins,
)
from vila_u.utils.hybrid_attention import (
    build_action_token_position_mask,
    build_hybrid_attention_mask,
)

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA-U-CoT-VLA"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class CoTVLAArguments:
    """Arguments for CoT-VLA training (Phase 4)"""
    data_root: str = field(
        default=None,
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
    use_discrete_action_prediction: bool = field(
        default=True,
        metadata={"help": "Use autoregressive discrete action tokens"}
    )
    use_hybrid_attention: bool = field(
        default=True,
        metadata={"help": "Use full attention inside the action token block"}
    )
    # Phase 4 specific
    use_visual_cot: bool = field(
        default=True,
        metadata={"help": "Enable visual chain-of-thought reasoning"}
    )
    subgoal_horizon_low: int = field(
        default=4,
        metadata={"help": "Minimum subgoal horizon (frames)"}
    )
    subgoal_horizon_high: int = field(
        default=16,
        metadata={"help": "Maximum subgoal horizon (frames)"}
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


def register_phase4_tokens(tokenizer, model):
    """Register Phase 4 special tokens for visual CoT"""
    phase4_tokens = {
        "additional_special_tokens": [
            DEFAULT_ACT_START_TOKEN,
            DEFAULT_ACT_END_TOKEN,
            DEFAULT_SUBGOAL_START_TOKEN,
            DEFAULT_SUBGOAL_END_TOKEN,
        ]
    }
    smart_tokenizer_and_embedding_resize(
        special_tokens_dict=phase4_tokens,
        tokenizer=tokenizer,
        model=model.llm,
    )
    mprint(f"Registered Phase 4 tokens: {phase4_tokens['additional_special_tokens']}")


class CoTVLADataCollator:
    """Data collator for Phase 4 visual CoT training"""
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
        vision_tower=None,
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.action_token_ids = action_token_ids
        self.action_slot_token_id = action_slot_token_id
        self.num_action_tokens = action_chunk_size * action_dim
        self.use_hybrid_attention = use_hybrid_attention
        self.vision_tower = vision_tower

    def __call__(self, batch):
        # Stack observation images
        images = torch.stack([item["observations"] for item in batch])

        # Encode subgoal images to token IDs using vision tower
        if "subgoal_images" in batch[0]:
            subgoal_images = torch.stack([item["subgoal_images"] for item in batch])

            # Encode subgoal images to codebook indices (token IDs)
            with torch.no_grad():
                # Move to same device as vision tower
                device = next(self.vision_tower.parameters()).device
                subgoal_images = subgoal_images.to(device)

                # Encode: returns (code, z_q) where code is [B, H, W, depth]
                code, _ = self.vision_tower.vision_tower.rqvaesiglip.encode_image(subgoal_images)

                # Flatten code to token sequence: [B, H, W, depth] -> [B, H*W*depth]
                B, H, W, depth = code.shape
                subgoal_token_ids = code.reshape(B, H * W * depth).cpu()  # [B, 1024]
        else:
            subgoal_token_ids = None

        input_id_list = []
        label_list = []
        for i, item in enumerate(batch):
            prompt_ids = item["prompt_ids"]
            action_token_ids = item["action_token_ids"]

            # Construct input sequence
            if subgoal_token_ids is not None:
                # Phase 4: [prompt] [subgoal_tokens] [action_tokens]
                subgoal_ids = subgoal_token_ids[i]

                if self.use_hybrid_attention:
                    action_input_ids = torch.full_like(
                        action_token_ids,
                        fill_value=self.action_slot_token_id,
                    )
                else:
                    action_input_ids = action_token_ids

                input_ids = torch.cat([prompt_ids, subgoal_ids, action_input_ids], dim=0)

                # Labels: ignore prompt, predict subgoal tokens and action tokens
                labels = torch.cat(
                    [
                        torch.full_like(prompt_ids, IGNORE_INDEX),
                        subgoal_ids,  # Visual autoregressive loss
                        action_token_ids,   # Action prediction loss
                    ],
                    dim=0,
                )
            else:
                # Phase 2/3: [prompt] [action_tokens]
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

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "labels": labels,
        }


class CoTVLATrainer(VILAUTrainer):
    """Trainer for Phase 4 visual CoT-VLA"""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        core_model = model
        while hasattr(core_model, "module"):
            core_model = core_model.module

        # Phase 4: Visual CoT training
        if getattr(core_model.config, "use_visual_cot", False):
            # Prepare inputs with both observation and subgoal images
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

            # Build hybrid attention mask for action tokens
            if getattr(core_model.config, "use_hybrid_attention", False):
                hybrid_attention_mask = build_hybrid_attention_mask(
                    mm_attention_mask,
                    num_action_tokens=core_model.config.action_chunk_size * core_model.config.action_dim,
                    dtype=inputs_embeds.dtype,
                )
                action_token_count = (
                    core_model.config.action_chunk_size * core_model.config.action_dim
                )
                use_flash_hybrid = (
                    getattr(core_model.llm.config, "_attn_implementation", None)
                    == "flash_attention_2"
                )

                outputs = core_model.llm.model(
                    input_ids=None,
                    attention_mask=mm_attention_mask if use_flash_hybrid else hybrid_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                    seqlens_in_batch=mm_attention_mask.sum(dim=-1, dtype=torch.int32) if use_flash_hybrid else None,
                    num_action_tokens=action_token_count if use_flash_hybrid else None,
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

                # Compute action logits
                action_logits = core_model.llm.lm_head(action_hidden_states)
                selected_logits = compute_selected_token_logits(
                    action_logits,
                    self.action_token_ids,
                )

                # Compute visual autoregressive loss (for subgoal tokens)
                lm_logits = core_model.llm.lm_head(outputs.last_hidden_state)
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Separate visual and action losses
                # This is a simplified version - actual implementation needs to identify
                # subgoal token positions vs action token positions
                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

                return (loss, outputs) if return_outputs else loss
            else:
                # Without hybrid attention
                outputs = core_model.llm(
                    input_ids=None,
                    attention_mask=mm_attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    inputs_embeds=inputs_embeds,
                    labels=mm_labels,
                    use_cache=False,
                    output_attentions=False,
                    output_hidden_states=False,
                    return_dict=True,
                )
                return (outputs.loss, outputs) if return_outputs else outputs.loss

        # Fallback to standard training
        return super().compute_loss(model, inputs, return_outputs, **kwargs)


def make_cot_vla_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: CoTVLAArguments,
    image_processor,
    training_args: TrainingArguments,
    mm_use_im_start_end: bool,
    action_token_ids,
    action_slot_token_id: int | None,
    vision_tower=None,
) -> Dict:
    """Create data module for CoT-VLA training"""
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    train_dataset = LiberoGoalDataset(
        data_root=data_args.data_root,
        image_processor=image_processor,
        tokenizer=tokenizer,
        action_chunk_size=data_args.action_chunk_size,
        image_size=data_args.image_size,
        remove_pause_intervals=data_args.remove_pause_intervals,
        pause_threshold=data_args.pause_threshold,
        mm_use_im_start_end=mm_use_im_start_end,
        action_token_ids=action_token_ids,
        use_discrete_action_prediction=data_args.use_discrete_action_prediction,
        # Phase 4 specific
        use_visual_cot=data_args.use_visual_cot,
        subgoal_horizon_low=data_args.subgoal_horizon_low,
        subgoal_horizon_high=data_args.subgoal_horizon_high,
    )
    training_args.sample_lens = [len(train_dataset)]

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=CoTVLADataCollator(
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
            mm_use_im_start_end=mm_use_im_start_end,
            action_token_ids=action_token_ids,
            action_slot_token_id=action_slot_token_id,
            action_chunk_size=data_args.action_chunk_size,
            action_dim=data_args.action_dim,
            use_hybrid_attention=data_args.use_hybrid_attention,
            vision_tower=vision_tower,
        ),
    )


def train():
    global local_rank

    from transformers import HfArgumentParser, AutoConfig, set_seed

    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        CoTVLAArguments
    ))
    model_args, data_args, training_args, cot_args = cast(
        Tuple[ModelArguments, DataArguments, TrainingArguments, CoTVLAArguments],
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
        # Set both resume_path and _name_or_path to ensure correct LLM path resolution
        config.resume_path = model_args.model_name_or_path
        config._name_or_path = model_args.model_name_or_path

    # Phase 4: Set CoT-VLA configuration
    config.use_visual_cot = cot_args.use_visual_cot
    config.subgoal_horizon_low = cot_args.subgoal_horizon_low
    config.subgoal_horizon_high = cot_args.subgoal_horizon_high
    config.use_discrete_action_prediction = cot_args.use_discrete_action_prediction
    config.use_hybrid_attention = cot_args.use_hybrid_attention
    config.action_chunk_size = cot_args.action_chunk_size
    config.action_dim = cot_args.action_dim
    config.action_num_bins = ACTION_NUM_BINS

    prepare_config_for_training(config, model_args, training_args, data_args)

    mprint(f"Loading model from {model_args.model_name_or_path}")

    # Determine attention implementation
    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2")
    if cot_args.use_hybrid_attention and attn_implementation == "flash_attention_2":
        attn_implementation = "eager"
    low_cpu_mem_usage = env_flag("LOW_CPU_MEM_USAGE", True)

    model = model_cls(
        config=config,
        attn_implementation=attn_implementation,
        model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    tokenizer = model.tokenizer

    # Register Phase 4 special tokens
    if cot_args.use_visual_cot:
        register_phase4_tokens(tokenizer, model)

    # Setup tokenizer padding
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
        raise ValueError("CoT-VLA training requires a vision tower")

    vision_tower.to(dtype=compute_dtype, device=training_args.device)

    # Setup action tokens
    action_token_ids = select_action_token_ids(tokenizer, ACTION_NUM_BINS)
    action_slot_token_id = None
    if cot_args.use_hybrid_attention:
        action_slot_token = "<action_slot>"
        tokenizer.add_tokens([action_slot_token], special_tokens=True)
        model.llm.resize_token_embeddings(len(tokenizer))
        action_slot_token_id = tokenizer.convert_tokens_to_ids(action_slot_token)

    # Create data module
    data_module = make_cot_vla_data_module(
        tokenizer=tokenizer,
        data_args=cot_args,
        image_processor=vision_tower.image_processor,
        training_args=training_args,
        mm_use_im_start_end=getattr(config, "mm_use_im_start_end", False),
        action_token_ids=action_token_ids,
        action_slot_token_id=action_slot_token_id,
        vision_tower=vision_tower,
    )

    # Store action token IDs in trainer for loss computation
    CoTVLATrainer.action_token_ids = action_token_ids

    # Create trainer
    trainer = CoTVLATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Add auto-resume callback
    trainer.add_callback(AutoResumeCallback(training_args.output_dir))

    # Start training
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    trainer.save_state()

    model.config.use_cache = True
    training_args.distributed_state.wait_for_everyone()

    if training_args.local_rank in (0, -1):
        trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()

