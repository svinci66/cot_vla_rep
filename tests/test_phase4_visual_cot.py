"""Phase 4 Visual CoT lightweight tests."""

import math
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch

from scripts.eval_oracle_subgoal_action_ablation_offline import (
    comparison_summary,
    parse_modes,
    result_mode_name,
    select_wrong_subgoal_indices,
)
from vila_u.constants import IGNORE_INDEX
from vila_u.train.train_action_prediction_main import (
    compute_visual_cot_loss,
    insert_subgoal_embeds_before_action_block,
    validate_visual_change_weighting,
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


def test_subgoal_timestep_uses_filtered_action_offset():
    from vila_u.data.libero_dataset_v2 import LiberoGoalDataset

    sample = {
        "timestep": 10,
        "filtered_timestep": 2,
        "non_pause_indices": [0, 4, 10, 20, 35],
        "num_frames": 40,
    }

    dataset = object.__new__(LiberoGoalDataset)
    dataset.remove_pause_intervals = True
    dataset.subgoal_sampling_strategy = "fixed"
    dataset.subgoal_min_offset = 1

    dataset.subgoal_max_offset = 1
    assert dataset._sample_subgoal_timestep(sample) == 20

    dataset.subgoal_max_offset = 2
    assert dataset._sample_subgoal_timestep(sample) == 35

    dataset.subgoal_max_offset = 10
    assert dataset._sample_subgoal_timestep(sample) == 35

    dataset.remove_pause_intervals = False
    dataset.subgoal_max_offset = 2
    assert dataset._sample_subgoal_timestep(sample) == 12

    dataset.subgoal_sampling_strategy = "uniform"
    dataset.subgoal_min_offset = 1
    dataset.subgoal_max_offset = 2
    observed = {dataset._sample_subgoal_timestep(sample) for _ in range(50)}
    assert observed.issubset({11, 12})
    assert observed
    print("✓ subgoal timestep uses filtered action-step offsets")


class FakeRQVAE(torch.nn.Module):
    def __init__(self, codes):
        super().__init__()
        self.codes = codes
        self.anchor = torch.nn.Parameter(torch.zeros(()))

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


def test_compute_visual_cot_loss_with_offset_codes():
    codes = torch.tensor([[[[0, 1], [2, 3]], [[1, 2], [3, 4]]]], dtype=torch.long)
    image_tokens = 4
    depth = 2
    vocab_size = 8
    hidden_size = 6
    offset = 32000
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size)
    subgoal_images = torch.randn(1, 3, 8, 8)
    core_model = FakeCoreModel(FakeVisionTower(codes, vocab_size))

    loss = compute_visual_cot_loss(
        core_model,
        subgoal_hidden_states,
        subgoal_images,
        subgoal_codes=codes.reshape(1, image_tokens, depth) + offset,
        subgoal_code_offset=offset,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() < 0.01
    print("✓ visual CoT loss offset-code path verified")


def test_compute_visual_cot_loss_with_change_weight():
    subgoal_codes = torch.tensor([[[[0], [1]], [[2], [3]]]], dtype=torch.long)
    current_codes = torch.tensor([[[[0], [4]], [[2], [5]]]], dtype=torch.long)
    image_tokens = 4
    depth = 1
    vocab_size = 8
    hidden_size = 6
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size)
    subgoal_images = torch.randn(1, 3, 8, 8)
    core_model = FakeCoreModel(FakeVisionTower(subgoal_codes, vocab_size))

    loss = compute_visual_cot_loss(
        core_model,
        subgoal_hidden_states,
        subgoal_images,
        subgoal_codes=subgoal_codes.reshape(1, image_tokens, depth),
        current_codes=current_codes.reshape(1, image_tokens, depth),
        use_change_weight=True,
        change_weight=3.0,
    )

    visual_logits = core_model.get_vision_tower().vision_tower.rqtransformer(
        subgoal_hidden_states,
        subgoal_codes.reshape(1, image_tokens, depth),
        core_model.get_vision_tower().vision_tower.rqvaesiglip,
    )
    per_token_loss = torch.nn.functional.cross_entropy(
        visual_logits.reshape(image_tokens * depth, vocab_size),
        subgoal_codes.reshape(image_tokens * depth),
        reduction="none",
    ).view(1, image_tokens, depth)
    changed_positions = current_codes.reshape(1, image_tokens, depth).ne(
        subgoal_codes.reshape(1, image_tokens, depth)
    ).any(dim=-1)
    weights = torch.where(
        changed_positions.unsqueeze(-1),
        torch.full_like(per_token_loss, 3.0),
        torch.ones_like(per_token_loss),
    )
    expected_loss = (per_token_loss * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected_loss)
    print("✓ visual CoT change-aware CE weighting verified")


def test_compute_visual_cot_loss_with_dynamic_change_weight():
    subgoal_codes = torch.tensor(
        [[[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]],
        dtype=torch.long,
    )
    current_codes = torch.tensor(
        [[[[0, 1, 2, 3], [16, 5, 6, 7]], [[17, 18, 10, 11], [19, 20, 21, 22]]]],
        dtype=torch.long,
    )
    image_tokens = 4
    depth = 4
    vocab_size = 32
    hidden_size = 6
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size)
    subgoal_images = torch.randn(1, 3, 8, 8)
    core_model = FakeCoreModel(FakeVisionTower(subgoal_codes, vocab_size))

    loss, stats = compute_visual_cot_loss(
        core_model,
        subgoal_hidden_states,
        subgoal_images,
        subgoal_codes=subgoal_codes.reshape(1, image_tokens, depth),
        current_codes=current_codes.reshape(1, image_tokens, depth),
        use_change_weight=True,
        change_weight=6.0,
        change_weight_mode="dynamic",
        unchanged_weight=0.5,
        return_stats=True,
    )

    visual_logits = core_model.get_vision_tower().vision_tower.rqtransformer(
        subgoal_hidden_states,
        subgoal_codes.reshape(1, image_tokens, depth),
        core_model.get_vision_tower().vision_tower.rqvaesiglip,
    )
    per_token_loss = torch.nn.functional.cross_entropy(
        visual_logits.reshape(image_tokens * depth, vocab_size),
        subgoal_codes.reshape(image_tokens * depth),
        reduction="none",
    ).view(1, image_tokens, depth)
    code_changed = current_codes.reshape(1, image_tokens, depth).ne(
        subgoal_codes.reshape(1, image_tokens, depth)
    )
    change_intensity = code_changed.float().mean(dim=-1)
    patch_weights = 0.5 + change_intensity * (6.0 - 0.5)
    expected_patch_weights = torch.tensor([[0.5, 1.875, 3.25, 6.0]])
    assert torch.allclose(patch_weights, expected_patch_weights)
    weights = patch_weights.unsqueeze(-1).expand_as(per_token_loss)
    expected_loss = (per_token_loss * weights).sum() / weights.sum()

    assert torch.allclose(loss, expected_loss)
    changed_positions = code_changed.any(dim=-1)
    expected_changed_weight_ratio = patch_weights[changed_positions].sum() / patch_weights.sum()
    assert torch.allclose(stats["visual_changed_patch_ratio"], changed_positions.float().mean())
    assert torch.allclose(
        stats["visual_effective_changed_weight_ratio"],
        expected_changed_weight_ratio,
    )
    assert torch.allclose(stats["visual_mean_patch_weight"], patch_weights.mean())
    print("✓ visual CoT dynamic change-aware CE weighting verified")


def test_compute_visual_cot_loss_with_dynamic_change_threshold():
    subgoal_codes = torch.tensor(
        [[[[0, 1, 2, 3], [4, 5, 6, 7]], [[8, 9, 10, 11], [12, 13, 14, 15]]]],
        dtype=torch.long,
    )
    current_codes = torch.tensor(
        [[[[0, 1, 2, 3], [16, 5, 6, 7]], [[17, 18, 10, 11], [19, 20, 21, 22]]]],
        dtype=torch.long,
    )
    image_tokens = 4
    depth = 4
    vocab_size = 32
    hidden_size = 6
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size)
    subgoal_images = torch.randn(1, 3, 8, 8)
    core_model = FakeCoreModel(FakeVisionTower(subgoal_codes, vocab_size))

    loss, stats = compute_visual_cot_loss(
        core_model,
        subgoal_hidden_states,
        subgoal_images,
        subgoal_codes=subgoal_codes.reshape(1, image_tokens, depth),
        current_codes=current_codes.reshape(1, image_tokens, depth),
        use_change_weight=True,
        change_weight=6.0,
        change_weight_mode="dynamic",
        change_intensity_threshold=0.25,
        unchanged_weight=0.5,
        return_stats=True,
    )

    visual_logits = core_model.get_vision_tower().vision_tower.rqtransformer(
        subgoal_hidden_states,
        subgoal_codes.reshape(1, image_tokens, depth),
        core_model.get_vision_tower().vision_tower.rqvaesiglip,
    )
    per_token_loss = torch.nn.functional.cross_entropy(
        visual_logits.reshape(image_tokens * depth, vocab_size),
        subgoal_codes.reshape(image_tokens * depth),
        reduction="none",
    ).view(1, image_tokens, depth)
    code_changed = current_codes.reshape(1, image_tokens, depth).ne(
        subgoal_codes.reshape(1, image_tokens, depth)
    )
    raw_intensity = code_changed.float().mean(dim=-1)
    effective_intensity = ((raw_intensity - 0.25) / 0.75).clamp(min=0.0, max=1.0)
    patch_weights = 0.5 + effective_intensity * (6.0 - 0.5)
    expected_patch_weights = torch.tensor([[0.5, 0.5, 2.3333333, 6.0]])

    assert torch.allclose(patch_weights, expected_patch_weights)
    weights = patch_weights.unsqueeze(-1).expand_as(per_token_loss)
    expected_loss = (per_token_loss * weights).sum() / weights.sum()
    assert torch.allclose(loss, expected_loss)
    assert torch.allclose(stats["visual_mean_raw_change_intensity"], raw_intensity.mean())
    assert torch.allclose(
        stats["visual_mean_effective_change_intensity"],
        effective_intensity.mean(),
    )
    assert torch.allclose(
        stats["visual_thresholded_changed_patch_ratio"],
        effective_intensity.gt(0).float().mean(),
    )
    print("✓ visual CoT dynamic change threshold verified")


def test_visual_change_weighting_validation():
    assert validate_visual_change_weighting("DYNAMIC", 0.25) == ("dynamic", 0.25)
    assert validate_visual_change_weighting("binary", 0.0) == ("binary", 0.0)

    for mode, threshold in (
        ("unsupported", 0.0),
        ("dynamic", -0.1),
        ("dynamic", 1.0),
        ("dynamic", math.inf),
        ("dynamic", math.nan),
    ):
        try:
            validate_visual_change_weighting(mode, threshold)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f"Expected invalid visual change configuration to fail: {mode=}, {threshold=}"
            )
    print("✓ visual change weighting validation verified")


def test_oracle_ablation_modes_follow_configured_offset():
    assert parse_modes("no_subgoal,oracle,wrong") == ["no_subgoal", "oracle", "wrong"]
    assert parse_modes("oracle_t10,wrong_t10,oracle") == ["oracle", "wrong"]
    assert result_mode_name("oracle", 5) == "oracle_t5"
    assert result_mode_name("wrong", 5) == "wrong_t5"

    summaries = {
        "oracle_t5": {"mae": 0.1},
        "wrong_t5": {"mae": 0.2},
        "no_subgoal": {"mae": 0.3},
    }
    comparisons = comparison_summary(summaries, subgoal_offset=5)
    assert comparisons["oracle_t5_minus_wrong_t5"]["mae"] < 0
    assert comparisons["oracle_t5_minus_no_subgoal"]["mae_improved"]
    print("✓ oracle ablation mode labels follow configured offset")


def test_wrong_subgoal_selection_avoids_same_demo():
    samples = [
        {"file": "task_a.hdf5", "demo": "demo_0"},
        {"file": "task_a.hdf5", "demo": "demo_0"},
        {"file": "task_a.hdf5", "demo": "demo_1"},
        {"file": "task_b.hdf5", "demo": "demo_0"},
        {"file": "task_b.hdf5", "demo": "demo_1"},
    ]
    wrong_indices = select_wrong_subgoal_indices(samples, [0, 1, 3], shift=1)
    for selected_index, wrong_index in zip([0, 1, 3], wrong_indices):
        assert samples[wrong_index]["file"] != samples[selected_index]["file"]

    one_task_samples = samples[:3]
    fallback_index = select_wrong_subgoal_indices(one_task_samples, [0], shift=0)[0]
    assert one_task_samples[fallback_index]["demo"] != one_task_samples[0]["demo"]

    try:
        select_wrong_subgoal_indices(one_task_samples[:2], [0], shift=0)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected same-demo-only negative selection to fail")
    print("✓ wrong subgoal selection avoids the source demonstration")


def test_dynamic_patch_weight_formula_covers_depth_counts():
    code_changed = torch.tensor(
        [
            [
                [False, False, False, False],
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, True],
            ]
        ]
    )
    change_intensity = code_changed.float().mean(dim=-1)
    patch_weights = 0.5 + change_intensity * (6.0 - 0.5)
    expected = torch.tensor([[0.5, 1.875, 3.25, 6.0]])

    assert torch.allclose(patch_weights, expected)
    print("✓ dynamic patch weight formula covers 0/4, 1/4, 2/4, 4/4 changes")


def test_depth_transformer_can_be_trainable_independently():
    rqvae = torch.nn.Linear(2, 2)
    rqtransformer = torch.nn.Linear(2, 2)

    rqvae.requires_grad_(False)
    rqtransformer.requires_grad_(True)

    assert not any(param.requires_grad for param in rqvae.parameters())
    assert all(param.requires_grad for param in rqtransformer.parameters())
    print("✓ depth transformer trainability can be independent from RQVAE")


def test_visual_cot_loss_updates_rqtransformer_parameters():
    class TrainableRQTransformer(torch.nn.Module):
        def __init__(self, hidden_size, vocab_size):
            super().__init__()
            self.proj = torch.nn.Linear(hidden_size, vocab_size)

        def forward(self, embed_from_body, code, model_aux=None):
            logits = self.proj(embed_from_body)
            return logits.unsqueeze(2).expand(-1, -1, code.shape[-1], -1)

    codes = torch.tensor([[[[0, 1], [2, 3]], [[1, 2], [3, 4]]]], dtype=torch.long)
    image_tokens = 4
    hidden_size = 6
    vocab_size = 8
    subgoal_hidden_states = torch.randn(1, image_tokens, hidden_size, requires_grad=True)
    subgoal_images = torch.randn(1, 3, 8, 8)
    vision_tower = FakeVisionTower(codes, vocab_size)
    vision_tower.vision_tower.rqvaesiglip.requires_grad_(False)
    vision_tower.vision_tower.rqtransformer = TrainableRQTransformer(hidden_size, vocab_size)
    core_model = FakeCoreModel(vision_tower)

    loss = compute_visual_cot_loss(core_model, subgoal_hidden_states, subgoal_images)
    loss.backward()

    grads = [
        param.grad
        for param in vision_tower.vision_tower.rqtransformer.parameters()
        if param.requires_grad
    ]
    assert grads
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    print("✓ visual CoT loss updates RQTransformer parameters")


def test_freeze_patch_keeps_depth_transformer_trainable():
    from vila_u.model.multimodal_encoder.rqvaesigliptransformer_encoder import (
        RQVAESIGLIPTransformerVisionTower,
    )

    class FakeMetaModel(torch.nn.Module):
        def __init__(self, vision_tower):
            super().__init__()
            self.vision_tower = vision_tower
            self.config = type(
                "Config",
                (),
                {
                    "tune_language_model": True,
                    "tune_vision_tower": False,
                    "tune_depth_transformer": True,
                    "tune_mm_projector": True,
                },
            )()

        def get_llm(self):
            return None

        def get_vision_tower(self):
            return self.vision_tower

        def get_mm_projector(self):
            return None

    class FakeRQVAESIGLIP(torch.nn.Module):
        pass

    class FakeRQTransformer(torch.nn.Module):
        pass

    class FakeInner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rqvaesiglip = FakeRQVAESIGLIP()
            self.rqtransformer = FakeRQTransformer()

    class FakeRQVisionTower(RQVAESIGLIPTransformerVisionTower):
        def __init__(self):
            torch.nn.Module.__init__(self)
            self.vision_tower = FakeInner()

    model = FakeMetaModel(FakeRQVisionTower())
    model.train()
    type(model).freezed_module_patch = __import__(
        "vila_u.model.vila_u_arch",
        fromlist=["VILAUMetaModel"],
    ).VILAUMetaModel.freezed_module_patch

    model.freezed_module_patch()

    assert not model.vision_tower.vision_tower.rqvaesiglip.training
    assert model.vision_tower.vision_tower.rqtransformer.training
    print("✓ freeze patch keeps depth transformer in train mode")


if __name__ == "__main__":
    test_causal_attention_mask_4d()
    test_insert_subgoal_embeds_before_action_block()
    test_subgoal_timestep_uses_filtered_action_offset()
    test_compute_visual_cot_loss_with_fake_rqtransformer()
    test_compute_visual_cot_loss_with_offset_codes()
    test_compute_visual_cot_loss_with_change_weight()
    test_compute_visual_cot_loss_with_dynamic_change_weight()
    test_compute_visual_cot_loss_with_dynamic_change_threshold()
    test_visual_change_weighting_validation()
    test_oracle_ablation_modes_follow_configured_offset()
    test_wrong_subgoal_selection_avoids_same_demo()
    test_dynamic_patch_weight_formula_covers_depth_counts()
    test_depth_transformer_can_be_trainable_independently()
    test_visual_cot_loss_updates_rqtransformer_parameters()
    test_freeze_patch_keeps_depth_transformer_trainable()
