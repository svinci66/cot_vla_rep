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
    """Build additive hybrid attention masks (optimized vectorized version).

    Default behavior is causal attention on valid tokens.
    The last ``num_action_tokens`` valid positions are treated as action slots and
    upgraded to full attention over all valid positions in the same sample.

    Optimized to avoid Python loops and CPU-GPU synchronization.
    """

    if attention_mask.ndim != 2:
        raise ValueError(
            f"Expected 2D attention mask of shape [B, L], got {attention_mask.shape}"
        )

    batch_size, seq_len = attention_mask.shape
    device = attention_mask.device
    mask_2d = attention_mask.bool()

    # Compute valid lengths for all samples at once (stays on GPU)
    valid_lens = mask_2d.sum(dim=1)  # [B]

    # Create base causal mask [L, L]
    causal_base = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    # Expand to batch: [B, L, L]
    allowed = causal_base.unsqueeze(0).expand(batch_size, -1, -1).clone()

    # Vectorized action token full attention
    # For each sample, action tokens (last num_action_tokens valid positions)
    # should attend to all valid positions
    if num_action_tokens > 0:
        # Create range tensors
        row_idx = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
        col_idx = torch.arange(seq_len, device=device).unsqueeze(1)  # [L, 1]

        # Compute action start positions: valid_len - num_action_tokens
        action_starts = (valid_lens - num_action_tokens).clamp(min=0).unsqueeze(1).unsqueeze(2)  # [B, 1, 1]
        valid_lens_expanded = valid_lens.unsqueeze(1).unsqueeze(2)  # [B, 1, 1]

        # Mask for action token rows: row_idx >= action_start and row_idx < valid_len
        is_action_row = (row_idx >= action_starts) & (row_idx < valid_lens_expanded)  # [B, L, 1]

        # Mask for valid columns: col_idx < valid_len
        is_valid_col = (col_idx < valid_lens_expanded)  # [B, 1, L]

        # Action tokens can attend to all valid positions
        action_attention = is_action_row & is_valid_col  # [B, L, L]
        allowed = allowed | action_attention

    # Handle padding: padding positions attend only to themselves
    # Create padding mask: positions >= valid_len
    is_padding = row_idx >= valid_lens.unsqueeze(1).unsqueeze(2)  # [B, L, 1]
    is_self = (row_idx.unsqueeze(0) == col_idx.unsqueeze(0))  # [1, L, L]
    padding_self_attention = is_padding & is_self
    allowed = allowed | padding_self_attention.squeeze(2).unsqueeze(1).expand(-1, seq_len, -1)

    # Convert to additive mask
    disallowed = torch.zeros(batch_size, 1, seq_len, seq_len, dtype=dtype, device=device)
    disallowed.masked_fill_(~allowed.unsqueeze(1), torch.finfo(dtype).min)

    return disallowed
