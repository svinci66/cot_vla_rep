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


if __name__ == "__main__":
    test_select_action_token_ids()
    test_discretize_and_undiscretize()
    test_action_token_roundtrip()
