"""
Phase 3 foundation tests: hybrid attention mask utilities.
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

import torch

from vila_u.utils.hybrid_attention import (
    build_action_token_position_mask,
    build_hybrid_attention_mask,
)


def test_action_token_position_mask():
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    action_mask = build_action_token_position_mask(attention_mask, num_action_tokens=2)

    expected = torch.tensor(
        [
            [0, 0, 0, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 1],
        ],
        dtype=torch.bool,
    )
    assert torch.equal(action_mask, expected)
    print("✓ action token position mask verified")


def test_hybrid_attention_mask():
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    hybrid_mask = build_hybrid_attention_mask(
        attention_mask,
        num_action_tokens=2,
        dtype=torch.float32,
    )

    matrix = hybrid_mask[0, 0]

    # Non-action token row keeps causal masking.
    assert matrix[1, 2] < 0
    assert matrix[2, 1] == 0

    # Action rows (last two valid positions) can see all valid positions.
    assert torch.all(matrix[3, :5] == 0)
    assert torch.all(matrix[4, :5] == 0)

    print("✓ hybrid attention mask verified")


if __name__ == "__main__":
    test_action_token_position_mask()
    test_hybrid_attention_mask()
