from __future__ import annotations

import torch


def build_action_token_position_mask(
    attention_mask: torch.Tensor,
    num_action_tokens: int,
) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention mask of shape [B, L], got {attention_mask.shape}"
        )

    mask_2d = attention_mask.bool()
    batch_size, seq_len = mask_2d.shape
    action_mask = torch.zeros_like(mask_2d, dtype=torch.bool)

    for batch_idx in range(batch_size):
        valid_len = int(mask_2d[batch_idx].sum().item())
        action_len = min(num_action_tokens, valid_len)
        if action_len > 0:
            action_mask[batch_idx, valid_len - action_len : valid_len] = True

    return action_mask


def build_hybrid_attention_mask(
    attention_mask: torch.Tensor,
    num_action_tokens: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build additive hybrid attention masks.

    Default behavior is causal attention on valid tokens.
    The last ``num_action_tokens`` valid positions are treated as action slots and
    upgraded to full attention over all valid positions in the same sample.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention mask of shape [B, L], got {attention_mask.shape}"
        )

    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    mask_2d = attention_mask.bool()

    disallowed = torch.full(
        (batch_size, 1, seq_len, seq_len),
        torch.finfo(dtype).min,
        dtype=dtype,
        device=device,
    )

    for batch_idx in range(batch_size):
        valid_len = int(mask_2d[batch_idx].sum().item())
        if valid_len == 0:
            disallowed[batch_idx, 0].fill_(0)
            continue

        allowed = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
        causal = torch.tril(torch.ones((valid_len, valid_len), dtype=torch.bool, device=device))
        allowed[:valid_len, :valid_len] = causal

        action_len = min(num_action_tokens, valid_len)
        if action_len > 0:
            action_start = valid_len - action_len
            allowed[action_start:valid_len, :valid_len] = True

        if valid_len < seq_len:
            pad_indices = torch.arange(valid_len, seq_len, device=device)
            allowed[pad_indices, pad_indices] = True

        disallowed[batch_idx, 0].masked_fill_(allowed, 0)

    return disallowed
