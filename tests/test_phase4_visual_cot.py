"""Phase 4 Visual CoT lightweight tests."""

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch

from vila_u.constants import IGNORE_INDEX
from vila_u.train.train_action_prediction_main import (
    compute_visual_cot_loss,
    insert_subgoal_embeds_before_action_block,
)
from vila_u.utils.hybrid_attention import build_causal_attention_mask


def test_causal_attention_mask_4d():
    attention_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)
    causal_mask = build_causal_attention_mask(attention_mask, dtype=torch.float32)

    assert tuple(causal_mask.shape) == (1, 1, 4, 4)
    matrix = causal_mask[0, 0]
    assert matrix[0, 0] == 0
    assert matrix[1, 0] == 0
    assert matrix[1, 2] < 0
    assert matrix[2, :3].eq(0).all()
    assert matrix[3, 3] == 0
    assert matrix[0, 3] < 0
    print("✓ causal 4D attention mask verified")


def test_insert_subgoal_embeds_before_action_block():
    batch_size = 2
    seq_len = 6
    hidden_size = 4
    label_depth = 1
    num_action_tokens = 2
    subgoal_len = 3

    inputs_embeds = torch.arange(
        batch_size * seq_len * hidden_size,
        dtype=torch.float32,
    ).view(batch_size, seq_len, hidden_size)
    labels = torch.full(
        (batch_size, seq_len, label_depth),
        IGNORE_INDEX,
        dtype=torch.long,
    )
    labels[0, 3:5, 0] = torch.tensor([101, 102])
    labels[1, 2:4, 0] = torch.tensor([201, 202])
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 0, 0],
        ],
        dtype=torch.bool,
    )
    position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)
    subgoal_embeds = torch.full(
        (batch_size, subgoal_len, hidden_size),
        10.0,
        dtype=torch.float32,
    )

    new_embeds, new_labels, new_attention_mask, new_position_ids, subgoal_prediction_mask = (
        insert_subgoal_embeds_before_action_block(
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            subgoal_embeds=subgoal_embeds,
            num_action_tokens=num_action_tokens,
        )
    )

    assert tuple(new_embeds.shape) == (2, 8, hidden_size)
    assert new_attention_mask[0].sum().item() == 8
    assert new_attention_mask[1].sum().item() == 7
    assert new_labels[0, 6:8, 0].tolist() == [101, 102]
    assert new_labels[1, 5:7, 0].tolist() == [201, 202]
    assert new_labels[0, 3:6, 0].eq(IGNORE_INDEX).all()
    assert subgoal_prediction_mask[0].sum().item() == subgoal_len
    assert subgoal_prediction_mask[1].sum().item() == subgoal_len
    assert new_position_ids[0, :8].tolist() == list(range(8))
    print("✓ subgoal insertion before action block verified")


class FakeRQVAE:
    def __init__(self, codes):
        self.codes = codes

    def encode_image(self, images):
        return self.codes.to(images.device), None


class FakeRQTransformer(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.vocab_size = vocab_size

    def forward(self, embed_from_body, code, model_aux=None):
        batch_size, seq_len, depth = code.shape
        logits = torch.full(
            (batch_size, seq_len, depth, self.vocab_size),
            -5.0,
            device=embed_from_body.device,
            dtype=embed_from_body.dtype,
        )
        logits.scatter_(-1, code.unsqueeze(-1).to(embed_from_body.device), 5.0)
        return logits


class FakeVisionModel:
    def __init__(self, codes, vocab_size):
        self.rqvaesiglip = FakeRQVAE(codes)
        self.rqtransformer = FakeRQTransformer(vocab_size)


class FakeVisionTower(torch.nn.Module):
    def __init__(self, codes, vocab_size):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vision_tower = FakeVisionModel(codes, vocab_size)


class FakeCoreModel:
    def __init__(self, vision_tower):
        self.vision_tower = vision_tower

    def get_vision_tower(self):
        return self.vision_tower


def test_compute_visual_cot_loss_with_fake_rqtransformer():
    codes = torch.tensor(
        [
            [
                [[0, 1], [2, 3]],
                [[1, 2], [3, 4]],
            ]
        ],
        dtype=torch.long,
    )
    image_tokens = 4
    depth = 2
    vocab_size = 8
    hidden_size = 6
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size)
    subgoal_images = torch.randn(1, 3, 8, 8)
    core_model = FakeCoreModel(FakeVisionTower(codes, vocab_size))

    loss = compute_visual_cot_loss(core_model, subgoal_hidden_states, subgoal_images)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() < 0.01
    print("✓ visual CoT loss shape verified")


if __name__ == "__main__":
    test_causal_attention_mask_4d()
    test_insert_subgoal_embeds_before_action_block()
    test_compute_visual_cot_loss_with_fake_rqtransformer()
