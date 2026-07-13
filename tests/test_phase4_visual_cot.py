"""Phase 4 Visual CoT lightweight tests."""

import math
import os
import sys
from types import SimpleNamespace

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch

from scripts.eval_oracle_subgoal_action_ablation_offline import (
    VALID_MODES,
    build_mode_items,
    comparison_summary,
    parse_modes,
    predict_training_batch_with_optional_subgoal,
    select_balanced_sample_indices,
    select_negative_subgoal_indices,
    validate_prediction_outputs,
)
from vila_u.constants import ACTION_NUM_BINS, IGNORE_INDEX
from vila_u.model.configuration_vila_u import VILAUConfig
from vila_u.train.train_action_prediction_main import (
    compute_visual_cot_loss,
    insert_subgoal_embeds_before_action_block,
    resolve_checkpoint_action_bin_edges,
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
    try:
        dataset._sample_subgoal_timestep(sample)
    except IndexError:
        pass
    else:
        raise AssertionError("Subgoal sampling must not clamp to the final frame")

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


def test_oracle_ablation_modes_and_action_labels():
    assert parse_modes(",".join(VALID_MODES)) == list(VALID_MODES)
    assert parse_modes("oracle,wrong") == ["oracle_t10", "cross_task_wrong_t10"]

    action_labels = torch.randn(10, 7)
    observation = torch.randn(3, 8, 8)
    oracle = torch.randn(3, 8, 8)
    source_item = {
        "observations": observation,
        "subgoal_images": oracle,
        "subgoal_timestep": 20,
        "subgoal_filtered_timestep": 10,
        "action_labels": action_labels,
        "timestep": 2,
        "filtered_timestep": 0,
    }
    same_negative = {
        "subgoal_images": torch.randn(3, 8, 8),
        "subgoal_timestep": 40,
        "subgoal_filtered_timestep": 30,
    }
    cross_negative = {
        "subgoal_images": torch.randn(3, 8, 8),
        "subgoal_timestep": 50,
        "subgoal_filtered_timestep": 35,
    }
    for mode in VALID_MODES:
        mode_item = build_mode_items(
            mode,
            [source_item],
            [same_negative],
            [cross_negative],
        )[0]
        assert torch.equal(mode_item["action_labels"], action_labels)
        if mode == "no_subgoal":
            assert "subgoal_images" not in mode_item
        elif mode == "current_copy":
            assert torch.equal(mode_item["subgoal_images"], observation)

    summaries = {mode: {"mae": index / 10.0} for index, mode in enumerate(VALID_MODES)}
    comparisons = comparison_summary(summaries)
    assert "oracle_t10_minus_cross_task_wrong_t10" in comparisons
    print("✓ five ablation modes preserve action labels and current_copy pixels")


def test_strong_negative_selection_and_balanced_samples():
    source_samples = [
        {"file": "task_a.hdf5", "demo": "demo_40", "filtered_timestep": 5},
        {"file": "task_b.hdf5", "demo": "demo_40", "filtered_timestep": 7},
    ]
    negative_samples = [
        {"file": "task_a.hdf5", "demo": "demo_41", "filtered_timestep": 10},
        {"file": "task_a.hdf5", "demo": "demo_42", "filtered_timestep": 30},
        {"file": "task_b.hdf5", "demo": "demo_41", "filtered_timestep": 8},
        {"file": "task_c.hdf5", "demo": "demo_40", "filtered_timestep": 5},
    ]
    same_index = select_negative_subgoal_indices(
        source_samples,
        [0],
        negative_samples,
        selection="same_task",
        seed=0,
        same_task_min_filtered_distance=20,
    )[0]
    assert negative_samples[same_index]["file"] == source_samples[0]["file"]
    assert negative_samples[same_index]["demo"] != source_samples[0]["demo"]
    assert abs(negative_samples[same_index]["filtered_timestep"] - 5) >= 20

    cross_indices = select_negative_subgoal_indices(
        source_samples,
        [0, 1],
        negative_samples,
        selection="cross_task",
        seed=3,
        same_task_min_filtered_distance=20,
    )
    for source_index, negative_index in zip([0, 1], cross_indices):
        assert negative_samples[negative_index]["file"] != source_samples[source_index]["file"]

    balanced_samples = [
        {"file": task_file, "demo": "demo_40", "filtered_timestep": index}
        for task_file in ("task_a.hdf5", "task_b.hdf5")
        for index in range(600)
    ]
    selected = select_balanced_sample_indices(balanced_samples, max_samples_per_task=500)
    assert len(selected) == 1000
    assert sum(balanced_samples[index]["file"] == "task_a.hdf5" for index in selected) == 500
    assert sum(balanced_samples[index]["file"] == "task_b.hdf5" for index in selected) == 500
    print("✓ negative pools are strong and per-task sampling is balanced")


def test_checkpoint_action_bins_are_reused_exactly():
    checkpoint_edges = torch.linspace(-1.0, 1.0, 7 * (ACTION_NUM_BINS + 1)).view(
        7,
        ACTION_NUM_BINS + 1,
    )
    config = SimpleNamespace(action_bin_edges=checkpoint_edges.tolist())
    action_args = SimpleNamespace(
        use_discrete_action_prediction=True,
        use_action_percentile_bins=True,
        require_checkpoint_action_bin_edges=True,
        action_dim=7,
    )
    resolved = resolve_checkpoint_action_bin_edges(config, action_args)
    assert torch.equal(resolved, checkpoint_edges)
    saved_config = VILAUConfig(
        demo_start_index=0,
        demo_end_index=40,
        require_checkpoint_action_bin_edges=True,
    )
    assert saved_config.demo_start_index == 0
    assert saved_config.demo_end_index == 40
    assert saved_config.require_checkpoint_action_bin_edges
    print("✓ checkpoint action_bin_edges are reused without recomputation")


def test_two_batch_prediction_contract_for_all_modes():
    class FakeBackbone(torch.nn.Module):
        def forward(self, inputs_embeds=None, **kwargs):
            return SimpleNamespace(last_hidden_state=inputs_embeds)

    class FakeEvaluationModel:
        def __init__(self):
            self.config = SimpleNamespace(
                action_chunk_size=10,
                action_dim=7,
                use_hybrid_attention=True,
                action_token_ids=list(range(ACTION_NUM_BINS)),
            )
            self.llm = SimpleNamespace(
                model=FakeBackbone(),
                lm_head=torch.nn.Linear(8, ACTION_NUM_BINS, bias=False),
            )

        def prepare_inputs_labels_for_multimodal(
            self,
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images,
        ):
            batch_size = input_ids.shape[0]
            sequence_length = 72
            inputs_embeds = torch.zeros(batch_size, sequence_length, 8)
            mm_labels = torch.full((batch_size, sequence_length, 1), IGNORE_INDEX)
            mm_labels[:, -70:, 0] = 0
            mm_attention_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
            position_ids = torch.arange(sequence_length).unsqueeze(0).repeat(batch_size, 1)
            return (
                None,
                position_ids,
                mm_attention_mask,
                None,
                inputs_embeds,
                mm_labels,
            )

        def encode_images(self, images, image_ids=None):
            return torch.zeros(images.shape[0], 2, 8), None

    model = FakeEvaluationModel()
    for batch_size in (2, 1):
        source_items = [
            {
                "observations": torch.zeros(3, 8, 8),
                "subgoal_images": torch.ones(3, 8, 8),
                "subgoal_timestep": 10,
                "subgoal_filtered_timestep": 10,
                "action_labels": torch.zeros(10, 7),
                "timestep": 0,
                "filtered_timestep": 0,
            }
            for _ in range(batch_size)
        ]
        negative_items = [
            {
                "subgoal_images": torch.full((3, 8, 8), 2.0),
                "subgoal_timestep": 30,
                "subgoal_filtered_timestep": 30,
            }
            for _ in range(batch_size)
        ]
        for mode in VALID_MODES:
            mode_items = build_mode_items(
                mode,
                source_items,
                negative_items,
                negative_items,
            )
            batch = {
                "input_ids": torch.zeros(batch_size, 72, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, 72, dtype=torch.bool),
                "labels": torch.zeros(batch_size, 72, dtype=torch.long),
                "images": torch.stack([item["observations"] for item in mode_items]),
            }
            if "subgoal_images" in mode_items[0]:
                batch["subgoal_images"] = torch.stack(
                    [item["subgoal_images"] for item in mode_items]
                )
            pred_bins, gt_bins = predict_training_batch_with_optional_subgoal(model, batch)
            assert tuple(gt_bins.shape) == (batch_size, 10, 7)
            pred_actions = pred_bins.float()
            assert torch.isfinite(pred_actions).float().mean().item() == 1.0
            validate_prediction_outputs(
                mode,
                pred_bins,
                pred_actions,
                batch_size=batch_size,
                action_chunk_size=10,
                action_dim=7,
            )
    print("✓ two batches satisfy the five-mode [B, 10, 7] finite contract")


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
    test_oracle_ablation_modes_and_action_labels()
    test_strong_negative_selection_and_balanced_samples()
    test_checkpoint_action_bins_are_reused_exactly()
    test_two_batch_prediction_contract_for_all_modes()
    test_dynamic_patch_weight_formula_covers_depth_counts()
    test_depth_transformer_can_be_trainable_independently()
    test_visual_cot_loss_updates_rqtransformer_parameters()
    test_freeze_patch_keeps_depth_transformer_trainable()
