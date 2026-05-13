"""
Phase 2 foundation tests: action discretization and tokenizer mapping.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch


class DummyTokenizer:
    def __init__(self):
        self._vocab = {f"tok_{i}": i for i in range(1000)}
        self.all_special_ids = [0, 1, 2]
        self._added_vocab = {"<image>": 998, "<im_start>": 999}

    def get_vocab(self):
        return self._vocab

    def get_added_vocab(self):
        return self._added_vocab


def test_select_action_token_ids():
    from vila_u.utils.action_tokenizer import select_action_token_ids

    tokenizer = DummyTokenizer()
    action_token_ids = select_action_token_ids(tokenizer, num_bins=8)

    assert action_token_ids == [990, 991, 992, 993, 994, 995, 996, 997]
    print("✓ action token IDs selected from tokenizer tail")


def test_discretize_and_undiscretize():
    from vila_u.utils.action_tokenizer import (
        discretize_actions,
        undiscretize_action_bins,
    )

    actions = torch.tensor([-1.0, -0.25, 0.0, 0.25, 1.0])
    bins = discretize_actions(actions)
    restored = undiscretize_action_bins(bins)

    assert bins.min() >= 0
    assert bins.max() <= 255
    assert restored.shape == actions.shape
    print("✓ discretization and restoration logic verified")


def test_action_token_roundtrip():
    from vila_u.utils.action_tokenizer import (
        actions_to_token_ids,
        token_ids_to_actions,
    )

    action_token_ids = list(range(32000, 32256))
    actions = torch.tensor([[-1.0, -0.5, 0.0, 0.5, 1.0]])

    token_ids = actions_to_token_ids(actions, action_token_ids)
    restored = token_ids_to_actions(token_ids, action_token_ids)

    assert token_ids.shape == actions.shape
    assert restored.shape == actions.shape
    assert torch.all(restored <= 1.0)
    assert torch.all(restored >= -1.0)
    print("✓ action/token roundtrip verified")


def test_allowed_action_token_logits_processor():
    from vila_u.utils.action_tokenizer import AllowedActionTokensLogitsProcessor

    processor = AllowedActionTokensLogitsProcessor([2, 4, 6])
    scores = torch.arange(8, dtype=torch.float32).unsqueeze(0)
    processed = processor(torch.tensor([[1, 2]]), scores.clone())

    allowed = {2, 4, 6}
    for idx in range(processed.shape[-1]):
        if idx in allowed:
            assert processed[0, idx] == scores[0, idx]
        else:
            assert torch.isneginf(processed[0, idx])
    print("✓ logits processor restricts generation to action tokens")


def test_compute_selected_token_logits():
    import torch.nn as nn
    from vila_u.utils.action_tokenizer import compute_selected_token_logits

    hidden_states = torch.randn(2, 3, 5)
    lm_head = nn.Linear(5, 12, bias=True)
    token_ids = [2, 4, 7, 9]

    full_logits = lm_head(hidden_states)
    selected_logits = compute_selected_token_logits(hidden_states, lm_head, token_ids)

    assert torch.allclose(selected_logits, full_logits[..., token_ids], atol=1e-5)
    print("✓ selected-token logits match sliced full-vocab logits")


def test_percentile_action_bin_edges_roundtrip():
    from vila_u.utils.action_tokenizer import (
        actions_to_token_ids,
        compute_percentile_action_bin_edges,
        discretize_actions,
        token_ids_to_actions,
    )

    actions = torch.tensor(
        [
            [-2.0, -0.5, 0.0],
            [-1.0, 0.0, 0.2],
            [0.0, 0.5, 0.4],
            [1.0, 1.0, 0.6],
            [2.0, 1.5, 0.8],
        ]
    )
    edges = compute_percentile_action_bin_edges(
        actions,
        num_bins=8,
        low_percentile=0.0,
        high_percentile=100.0,
    )
    bins = discretize_actions(actions, num_bins=8, bin_edges=edges)
    token_ids = actions_to_token_ids(actions, list(range(100, 108)), num_bins=8, bin_edges=edges)
    restored = token_ids_to_actions(token_ids, list(range(100, 108)), num_bins=8, bin_edges=edges)

    assert tuple(edges.shape) == (3, 9)
    assert bins.shape == actions.shape
    assert token_ids.shape == actions.shape
    assert restored.shape == actions.shape
    assert torch.all(restored[:, 0] >= edges[0, 0])
    assert torch.all(restored[:, 0] <= edges[0, -1])
    print("✓ percentile per-dim action bin edges roundtrip verified")


def test_percentile_token_decode_preserves_action_shape():
    from vila_u.utils.action_tokenizer import actions_to_token_ids, token_ids_to_actions

    actions = torch.tensor(
        [
            [
                [-1.0, 0.0, 1.0],
                [-0.5, 0.5, 0.75],
            ]
        ],
        dtype=torch.float32,
    )
    edges = torch.tensor(
        [
            [-1.0, -0.5, 0.0, 0.5, 1.0],
            [-1.0, -0.25, 0.25, 0.75, 1.0],
            [-1.0, 0.0, 0.5, 0.75, 1.0],
        ],
        dtype=torch.float32,
    )
    action_token_ids = list(range(200, 204))

    token_ids = actions_to_token_ids(actions, action_token_ids, num_bins=4, bin_edges=edges)
    restored = token_ids_to_actions(token_ids, action_token_ids, num_bins=4, bin_edges=edges)

    assert token_ids.shape == actions.shape
    assert restored.shape == actions.shape
    assert restored[0, 0, 0] != restored[0, 0, 1]
    print("✓ percentile token decoding preserves chunk/action-dim shape")


def test_config_stores_action_bin_edges():
    from vila_u.model.configuration_vila_u import VILAUConfig

    edges = [[-1.0, 0.0, 1.0], [-0.5, 0.5, 1.5]]
    config = VILAUConfig(
        action_num_bins=2,
        action_bin_edges=edges,
        use_action_percentile_bins=True,
        action_bin_low_percentile=1.0,
        action_bin_high_percentile=99.0,
    )

    assert config.action_bin_edges == edges
    assert config.use_action_percentile_bins is True
    assert config.action_bin_low_percentile == 1.0
    assert config.action_bin_high_percentile == 99.0
    print("✓ action bin edges stored in config")


def test_dedicated_action_slot_token_is_not_action_bin():
    from vila_u.constants import DEFAULT_ACTION_SLOT_TOKEN
    from vila_u.train.train_action_prediction_main import initialize_action_slot_token
    from vila_u.utils.action_tokenizer import select_action_token_ids

    class TinyTokenizer(DummyTokenizer):
        def __init__(self):
            super().__init__()
            self.unk_token_id = -1

        def convert_tokens_to_ids(self, token):
            return self._vocab.get(token, self.unk_token_id)

        def add_tokens(self, tokens, special_tokens=False):
            num_added = 0
            for token in tokens:
                if token not in self._vocab:
                    self._vocab[token] = max(self._vocab.values()) + 1
                    if special_tokens:
                        self._added_vocab[token] = self._vocab[token]
                    num_added += 1
            return num_added

    class TinyModel:
        def __init__(self):
            self.resize_calls = 0
            self.input_embeddings = torch.nn.Embedding(1001, 4)
            self.output_embeddings = torch.nn.Embedding(1001, 4)

        def resize_token_embeddings(self, vocab_size):
            self.resize_calls += 1
            self.input_embeddings = torch.nn.Embedding(vocab_size, 4)
            self.output_embeddings = torch.nn.Embedding(vocab_size, 4)

        def get_input_embeddings(self):
            return self.input_embeddings

        def get_output_embeddings(self):
            return self.output_embeddings

    tokenizer = TinyTokenizer()
    model = TinyModel()
    slot_id = initialize_action_slot_token(tokenizer, model)
    action_token_ids = select_action_token_ids(tokenizer, num_bins=8)

    assert tokenizer.convert_tokens_to_ids(DEFAULT_ACTION_SLOT_TOKEN) == slot_id
    assert slot_id not in action_token_ids
    assert model.resize_calls == 1
    print("✓ dedicated action slot token is separate from action bin tokens")


if __name__ == "__main__":
    test_select_action_token_ids()
    test_discretize_and_undiscretize()
    test_action_token_roundtrip()
    test_percentile_action_bin_edges_roundtrip()
    test_percentile_token_decode_preserves_action_shape()
    test_config_stores_action_bin_edges()
    test_dedicated_action_slot_token_is_not_action_bin()
    test_allowed_action_token_logits_processor()
    test_compute_selected_token_logits()
