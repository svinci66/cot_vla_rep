"""
VILA-U Action Prediction Training (Main)
Based on VILA-U's train.py framework with action prediction support
"""

import logging
import os
import pathlib
import torch
import transformers

from typing import Dict, Tuple, cast
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
    ACTION_NUM_BINS,
    IGNORE_INDEX,
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
    use_discrete_action_prediction: bool = field(
        default=True,
        metadata={"help": "Use autoregressive discrete action tokens instead of regression"}
    )
    use_hybrid_attention: bool = field(
        default=True,
        metadata={"help": "Use full attention inside the action token block"}
    )


class ActionPredictionDataCollator:
    def __init__(self, tokenizer, model_max_length: int, mm_use_im_start_end: bool):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end

    def __call__(self, batch):
        images = torch.stack([item["observations"] for item in batch])
        action_labels = torch.stack([item["action_labels"] for item in batch])
        input_ids = pad_sequence(
            [item["prompt_ids"] for item in batch],
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = input_ids[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "images": images,
            "action_labels": action_labels,
        }


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
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.action_token_ids = action_token_ids
        self.action_slot_token_id = action_slot_token_id
        self.num_action_tokens = action_chunk_size * action_dim
        self.use_hybrid_attention = use_hybrid_attention

    def __call__(self, batch):
        images = torch.stack([item["observations"] for item in batch])

        input_id_list = []
        label_list = []
        for item in batch:
            prompt_ids = item["prompt_ids"]
            action_token_ids = item["action_token_ids"]
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
                loss = torch.nn.functional.cross_entropy(
                    action_logits.reshape(-1, action_logits.size(-1)),
                    action_label_bins.reshape(-1),
                )
                if return_outputs:
                    return loss, {"logits": action_logits}
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

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


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


def make_action_prediction_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: ActionPredictionArguments,
    image_processor,
    training_args: TrainingArguments,
    mm_use_im_start_end: bool,
    action_token_ids,
    action_slot_token_id: int | None,
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
        mm_use_im_start_end=mm_use_im_start_end,
        action_token_ids=action_token_ids,
        use_discrete_action_prediction=data_args.use_discrete_action_prediction,
    )
    training_args.sample_lens = [len(train_dataset)]

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
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

    # Enable action prediction
    config.use_discrete_action_prediction = action_args.use_discrete_action_prediction
    config.use_action_prediction = not action_args.use_discrete_action_prediction
    config.use_hybrid_attention = action_args.use_hybrid_attention
    config.action_dim = action_args.action_dim
    config.action_chunk_size = action_args.action_chunk_size
    config.action_num_bins = ACTION_NUM_BINS
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
            model.get_vision_tower().vision_tower.rqvaesiglip.eval()
            model.get_vision_tower().vision_tower.rqtransformer.requires_grad_(
                training_args.tune_vision_tower
            )
            if not training_args.tune_vision_tower:
                model.get_vision_tower().vision_tower.rqtransformer.eval()
        else:
            raise NotImplementedError()
        print(f"vision tower {training_args.tune_vision_tower}")
        print(f"mm projector {training_args.tune_mm_projector}")

    # Action head is always trainable
    if hasattr(model, 'action_head'):
        model.action_head.requires_grad_(True)
        print(f"action head True")

    if not any([
        training_args.tune_language_model,
        training_args.tune_vision_tower,
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
        model.config.action_token_ids = action_token_ids
        model.config.action_num_bins = ACTION_NUM_BINS
        action_slot_token_id = action_token_ids[0]
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
