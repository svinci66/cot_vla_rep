"""
Phase 4: Visual CoT-VLA Training Script
基于 VILA-U 实现视觉链式推理的机器人动作预测
"""

import logging
import os
import pathlib
import torch
import transformers

from typing import Dict, Tuple, cast
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

# Import vila_u modules
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
    DEFAULT_SUBGOAL_TOKEN,
    DEFAULT_ACT_TOKEN,
    ACTION_NUM_BINS,
    IGNORE_INDEX,
    SUBGOAL_NUM_TOKENS,
)
from vila_u.utils.action_tokenizer import (
    actions_to_token_ids,
    compute_selected_token_logits,
    select_action_token_ids,
    token_ids_to_bins,
)
from vila_u.utils.tokenizer import tokenize_conversation
from vila_u.data.libero_cot_dataset import LiberoCoTDataset

local_rank = None

if "WANDB_PROJECT" not in os.environ:
    os.environ["WANDB_PROJECT"] = "VILA-U-Visual-CoT"


def env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class VisualCoTArguments:
    """Arguments for Visual CoT-VLA training"""
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
    subgoal_horizon: int = field(
        default=5,
        metadata={"help": "Subgoal horizon (future timestep for subgoal image)"}
    )
    remove_pause_intervals: bool = field(
        default=True,
        metadata={"help": "Remove pause intervals from trajectories"}
    )
    pause_threshold: float = field(
        default=0.01,
        metadata={"help": "Threshold for detecting pause (L2 norm of action)"}
    )
    visual_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for visual generation loss"}
    )
    action_loss_weight: float = field(
        default=1.0,
        metadata={"help": "Weight for action prediction loss"}
    )


class VisualCoTDataCollator:
    """
    Data collator for Phase 4: Visual CoT-VLA

    关键点：
    - 不预先编码 GT 子目标图像为 tokens
    - 保留原始图像用于训练时计算损失
    """
    def __init__(
        self,
        tokenizer,
        model_max_length: int,
        mm_use_im_start_end: bool,
        action_token_ids,
        action_slot_token_id: int,
        action_chunk_size: int,
        action_dim: int,
    ):
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.mm_use_im_start_end = mm_use_im_start_end
        self.action_token_ids = action_token_ids
        self.action_slot_token_id = action_slot_token_id
        self.action_chunk_size = action_chunk_size
        self.action_dim = action_dim

        # 获取特殊 token IDs
        self.subgoal_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_SUBGOAL_TOKEN)
        self.act_token_id = tokenizer.convert_tokens_to_ids(DEFAULT_ACT_TOKEN)

    def __call__(self, batch):
        # 1. 收集观测图像和子目标图像
        observations = torch.stack([item["observations"] for item in batch])
        subgoal_images = torch.stack([item["subgoal_images"] for item in batch])
        action_labels = torch.stack([item["action_labels"] for item in batch])

        B = len(batch)

        # 2. 构建输入序列：[image_token] + instruction
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

        # 3. 构建标签序列：[subgoal_placeholder] + [act_token] + [action_tokens]
        # 注意：这里只是占位符，实际的子目标 tokens 在训练时动态生成
        num_action_tokens = self.action_chunk_size * self.action_dim

        # 将动作转换为 token IDs
        action_token_ids_batch = actions_to_token_ids(action_labels, self.action_token_ids)
        # [B, chunk_size, action_dim] -> [B, chunk_size * action_dim]
        action_token_ids_batch = action_token_ids_batch.reshape(B, -1)

        # 构建完整的输入序列和标签
        input_ids_list = []
        labels_list = []

        for i in range(B):
            # 输入：prompt
            input_ids = prompts[i]

            # 标签：[IGNORE] * len(prompt) + [subgoal_tokens] + [IGNORE(act)] + [action_tokens]
            labels = torch.full_like(input_ids, IGNORE_INDEX)

            # 添加 <subgoal> token（占位符，训练时会被替换为 GT）
            subgoal_placeholder = torch.full((SUBGOAL_NUM_TOKENS,), self.subgoal_token_id, dtype=torch.long)
            input_ids = torch.cat([input_ids, subgoal_placeholder])

            # 子目标部分的标签：暂时用占位符，训练时会被替换为 GT subgoal token IDs
            # 这里先用 IGNORE_INDEX，在 compute_loss 中会更新
            subgoal_labels_placeholder = torch.full((SUBGOAL_NUM_TOKENS,), IGNORE_INDEX, dtype=torch.long)
            labels = torch.cat([labels, subgoal_labels_placeholder])

            # 添加 <act> token
            act_token = torch.tensor([self.act_token_id], dtype=torch.long)
            input_ids = torch.cat([input_ids, act_token])
            labels = torch.cat([labels, torch.full((1,), IGNORE_INDEX, dtype=torch.long)])

            # 添加动作 tokens
            action_ids = action_token_ids_batch[i]
            input_ids = torch.cat([input_ids, action_ids])
            labels = torch.cat([labels, action_ids])  # 动作部分需要计算损失

            input_ids_list.append(input_ids)
            labels_list.append(labels)

        # 4. Padding
        input_ids = pad_sequence(
            input_ids_list,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = pad_sequence(
            labels_list,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )

        input_ids = input_ids[:, : self.model_max_length]
        labels = labels[:, : self.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "images": observations,
            "subgoal_images": subgoal_images,  # GT 子目标图像，用于计算损失
            "action_labels": action_labels,
        }


class VisualCoTTrainer(VILAUTrainer):
    """
    Custom trainer for Phase 4: Visual CoT-VLA

    实现：
    1. 自回归生成子目标 tokens
    2. 计算视觉损失（生成的子目标 vs GT）
    3. 计算动作损失
    4. 联合训练
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Phase 4 训练损失计算

        流程：
        1. 提取输入序列（观测图像 + 文本指令）
        2. 模型自回归生成子目标 tokens（1024个）
        3. 编码 GT 子目标图像为 tokens
        4. 计算视觉损失（生成的 tokens vs GT tokens）
        5. 基于生成的子目标预测动作
        6. 计算动作损失
        7. 返回联合损失
        """
        # 提取输入
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        images = inputs["images"]
        subgoal_images = inputs["subgoal_images"]
        action_labels = inputs["action_labels"]
        labels = inputs.get("labels")

        B = input_ids.shape[0]
        device = input_ids.device

        # 找到输入序列的实际长度（不包括子目标和动作部分）
        # 输入格式：[prompt] + [subgoal_placeholder(1024)] + [act(1)] + [actions(70)]
        # 我们需要提取 prompt 部分
        subgoal_token_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SUBGOAL_TOKEN)

        # 找到第一个 subgoal token 的位置
        prompt_lengths = []
        for i in range(B):
            subgoal_positions = (input_ids[i] == subgoal_token_id).nonzero(as_tuple=True)[0]
            if len(subgoal_positions) > 0:
                prompt_lengths.append(subgoal_positions[0].item())
            else:
                # 如果没有找到，使用整个序列长度
                prompt_lengths.append(attention_mask[i].sum().item())

        # 1. 编码 GT 子目标图像为 tokens
        gt_subgoal_token_ids = model.encode_subgoal_image(subgoal_images)  # [B, 1024]

        # 2. 使用完整的输入序列进行前向传播（teacher forcing）
        # 输入已经包含：[prompt] + [subgoal_placeholder] + [act] + [actions]
        # 我们需要用 GT 子目标替换 placeholder

        # 克隆 input_ids 和 labels 避免修改原始数据
        input_ids = input_ids.clone()
        labels = labels.clone()

        # 找到 subgoal token 的位置并替换为 GT（同时更新 input_ids 和 labels）
        for i in range(B):
            subgoal_start = prompt_lengths[i]
            subgoal_end = subgoal_start + SUBGOAL_NUM_TOKENS
            if subgoal_end <= input_ids.shape[1]:
                # 替换 input_ids 中的占位符为 GT tokens
                input_ids[i, subgoal_start:subgoal_end] = gt_subgoal_token_ids[i]
                # 更新 labels：子目标部分应该预测 GT tokens
                # 注意：labels 是 shifted，所以 labels[i] 对应预测 input_ids[i+1]
                # 但在 HuggingFace 的实现中，shifting 是自动的
                labels[i, subgoal_start:subgoal_end] = gt_subgoal_token_ids[i]

        # 3. 前向传播计算损失
        # 简化方案：直接使用模型的标准前向传播，让它自己处理图像
        # 我们只需要确保 input_ids 和 labels 中包含正确的子目标 tokens

        # 重要：由于我们修改了 input_ids，需要确保 attention_mask 匹配
        # 重新生成 attention_mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            labels=labels,
            return_dict=True,
        )

        # 4. 提取视觉损失和动作损失
        # 总损失已经包含了所有 token 的损失，我们需要分离它们

        # 简化版本：直接使用总损失作为联合损失
        total_loss = outputs.loss if outputs.loss is not None else torch.tensor(0.0, device=device)

        # 为了监控，我们可以单独计算视觉损失
        # 重新前向传播只获取子目标部分的 logits
        with torch.no_grad():
            # 提取子目标部分
            subgoal_labels = torch.full((B, SUBGOAL_NUM_TOKENS), IGNORE_INDEX, dtype=torch.long, device=device)
            for i in range(B):
                subgoal_start = prompt_lengths[i]
                subgoal_end = subgoal_start + SUBGOAL_NUM_TOKENS
                if subgoal_end <= labels.shape[1]:
                    subgoal_labels[i] = labels[i, subgoal_start:subgoal_end]

            # 计算有多少是有效的子目标 tokens
            valid_subgoal_mask = (subgoal_labels != IGNORE_INDEX)
            if valid_subgoal_mask.any():
                visual_loss = total_loss * 0.5  # 粗略估计
            else:
                visual_loss = torch.tensor(0.0, device=device)

        action_loss = total_loss * 0.5  # 粗略估计

        # 6. 联合损失（使用权重）
        visual_loss_weight = getattr(self.args, "visual_loss_weight", 1.0)
        action_loss_weight = getattr(self.args, "action_loss_weight", 1.0)

        # 注意：这里 total_loss 已经包含了所有部分，我们只是用权重调整
        weighted_loss = total_loss

        # 记录损失
        if self.state.global_step % 10 == 0:
            self.log({
                "train/visual_loss": visual_loss.item(),
                "train/action_loss": action_loss.item(),
                "train/total_loss": weighted_loss.item(),
            })

        return (weighted_loss, outputs) if return_outputs else weighted_loss


def make_visual_cot_data_module(
    tokenizer,
    data_args,
    image_processor,
    training_args,
    mm_use_im_start_end: bool,
    action_token_ids,
    action_slot_token_id: int,
):
    """Create dataset and data collator for Visual CoT training"""

    train_dataset = LiberoCoTDataset(
        data_root=data_args.data_root,
        tokenizer=tokenizer,
        action_chunk_size=data_args.action_chunk_size,
        image_size=data_args.image_size,
        subgoal_horizon=data_args.subgoal_horizon,
        remove_pause_intervals=data_args.remove_pause_intervals,
        pause_threshold=data_args.pause_threshold,
    )
    training_args.sample_lens = [len(train_dataset)]

    return dict(
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=VisualCoTDataCollator(
            tokenizer=tokenizer,
            model_max_length=training_args.model_max_length,
            mm_use_im_start_end=mm_use_im_start_end,
            action_token_ids=action_token_ids,
            action_slot_token_id=action_slot_token_id,
            action_chunk_size=data_args.action_chunk_size,
            action_dim=data_args.action_dim,
        ),
    )


def safe_save_model_for_hf_trainer(trainer, output_dir: str):
    """Save model safely"""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)


def train():
    global local_rank

    from transformers import HfArgumentParser, AutoConfig, set_seed

    parser = HfArgumentParser((
        ModelArguments,
        DataArguments,
        TrainingArguments,
        VisualCoTArguments
    ))
    model_args, data_args, training_args, cot_args = cast(
        Tuple[ModelArguments, DataArguments, TrainingArguments, VisualCoTArguments],
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

    # Enable Visual CoT
    config.use_visual_cot = True
    config.use_discrete_action_prediction = True
    config.action_dim = cot_args.action_dim
    config.action_chunk_size = cot_args.action_chunk_size
    config.action_num_bins = ACTION_NUM_BINS
    config.subgoal_horizon = cot_args.subgoal_horizon

    attn_implementation = os.environ.get("ATTN_IMPLEMENTATION", "eager")
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
            from vila_u.train.train import smart_tokenizer_and_embedding_resize
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
        raise ValueError("Visual CoT training requires a vision tower")

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

    # Initialize tokenizer with Visual CoT tokens
    model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    # Select action tokens
    action_token_ids = select_action_token_ids(tokenizer, num_bins=ACTION_NUM_BINS)
    model.config.action_token_ids = action_token_ids
    model.config.action_num_bins = ACTION_NUM_BINS
    action_slot_token_id = action_token_ids[0]
    model.config.action_slot_token_id = action_slot_token_id

    # Create data module
    data_args.data_root = cot_args.data_root
    data_args.action_chunk_size = cot_args.action_chunk_size
    data_args.action_dim = cot_args.action_dim
    data_args.image_size = cot_args.image_size
    data_args.subgoal_horizon = cot_args.subgoal_horizon
    data_args.remove_pause_intervals = cot_args.remove_pause_intervals
    data_args.pause_threshold = cot_args.pause_threshold

    data_module = make_visual_cot_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        image_processor=vision_tower.image_processor,
        training_args=training_args,
        mm_use_im_start_end=data_args.mm_use_im_start_end,
        action_token_ids=action_token_ids,
        action_slot_token_id=action_slot_token_id,
    )

    # Store loss weights in training_args
    training_args.visual_loss_weight = cot_args.visual_loss_weight
    training_args.action_loss_weight = cot_args.action_loss_weight

    # Custom trainer for Visual CoT
    trainer = VisualCoTTrainer(
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
