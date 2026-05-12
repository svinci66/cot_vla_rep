from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F
from transformers import LogitsProcessor

from vila_u.constants import ACTION_MAX, ACTION_MIN, ACTION_NUM_BINS


@dataclass(frozen=True)
class ActionTokenSpec:
    token_ids: tuple[int, ...]
    num_bins: int = ACTION_NUM_BINS
    action_min: float = ACTION_MIN
    action_max: float = ACTION_MAX
    bin_edges: tuple[tuple[float, ...], ...] | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} action tokens, got {len(self.token_ids)}"
            )


class AllowedActionTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, action_token_ids: Sequence[int]):
        self.action_token_ids = tuple(action_token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        allowed_ids = torch.as_tensor(
            self.action_token_ids,
            dtype=torch.long,
            device=scores.device,
        )
        masked_scores = torch.full_like(scores, float("-inf"))
        masked_scores[:, allowed_ids] = scores[:, allowed_ids]
        return masked_scores


def compute_selected_token_logits(
    hidden_states: torch.Tensor,
    lm_head,
    token_ids: Sequence[int],
) -> torch.Tensor:
    token_tensor = torch.as_tensor(
        token_ids,
        dtype=torch.long,
        device=hidden_states.device,
    )
    weight = lm_head.weight.index_select(0, token_tensor).to(hidden_states.dtype)
    bias = None
    if getattr(lm_head, "bias", None) is not None:
        bias = lm_head.bias.index_select(0, token_tensor).to(hidden_states.dtype)
    return F.linear(hidden_states, weight, bias)


def select_action_token_ids(tokenizer, num_bins: int = ACTION_NUM_BINS) -> list[int]:
    """Select action tokens from the tail of the tokenizer vocabulary.

    The paper describes reusing low-frequency tokenizer tokens. Token usage
    frequency is not directly exposed by Hugging Face tokenizers, so this uses
    the standard tail-of-vocabulary heuristic: pick the highest-ID normal
    tokens while excluding special tokens and explicitly added tokens.
    """

    special_ids = set(getattr(tokenizer, "all_special_ids", []))
    added_vocab = getattr(tokenizer, "get_added_vocab", lambda: {})()
    added_token_ids = set(added_vocab.values())

    candidate_ids = sorted(set(tokenizer.get_vocab().values()))
    candidate_ids = [
        token_id
        for token_id in candidate_ids
        if token_id not in special_ids and token_id not in added_token_ids
    ]

    if len(candidate_ids) < num_bins:
        raise ValueError(
            f"Tokenizer only has {len(candidate_ids)} eligible tokens, need {num_bins}"
        )

    return candidate_ids[-num_bins:]


def normalize_action_bin_edges(
    bin_edges: torch.Tensor | Sequence[Sequence[float]] | None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    if bin_edges is None:
        return None
    edges = torch.as_tensor(bin_edges, dtype=dtype, device=device)
    if edges.ndim != 2:
        raise ValueError(f"Expected bin_edges with shape [action_dim, num_bins + 1], got {tuple(edges.shape)}")
    if edges.shape[1] < 2:
        raise ValueError("bin_edges must contain at least two boundaries per action dimension")
    if not torch.all(edges[:, 1:] >= edges[:, :-1]):
        raise ValueError("bin_edges must be monotonically non-decreasing per action dimension")
    return edges


def build_uniform_action_bin_edges(
    action_dim: int,
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    edges = torch.linspace(action_min, action_max, num_bins + 1, device=device, dtype=dtype)
    return edges.unsqueeze(0).repeat(action_dim, 1)


def compute_percentile_action_bin_edges(
    actions: torch.Tensor | Sequence[Sequence[float]],
    num_bins: int = ACTION_NUM_BINS,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    if actions.ndim < 2:
        raise ValueError(f"Expected actions with shape [..., action_dim], got {tuple(actions.shape)}")
    action_dim = actions.shape[-1]
    flat_actions = actions.reshape(-1, action_dim)
    lows = torch.quantile(flat_actions, low_percentile / 100.0, dim=0)
    highs = torch.quantile(flat_actions, high_percentile / 100.0, dim=0)
    too_small = (highs - lows).abs() < eps
    highs = torch.where(too_small, lows + eps, highs)
    scales = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float32).unsqueeze(0)
    return lows.unsqueeze(1) + (highs - lows).unsqueeze(1) * scales


def _reshape_edges_for_actions(edges: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    action_dim = actions.shape[-1]
    if edges.shape[0] == 1 and action_dim != 1:
        edges = edges.expand(action_dim, -1)
    if edges.shape[0] != action_dim:
        raise ValueError(
            f"bin_edges action_dim mismatch: edges have {edges.shape[0]} dims, actions have {action_dim}"
        )
    return edges


def discretize_actions(
    actions: torch.Tensor | Sequence[float],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
    bin_edges: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> torch.LongTensor:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    edges = normalize_action_bin_edges(bin_edges, device=actions.device, dtype=actions.dtype)
    if edges is None:
        clipped = actions.clamp(action_min, action_max)
        scale = (num_bins - 1) / (action_max - action_min)
        bins = torch.round((clipped - action_min) * scale).to(torch.long)
        return bins.clamp_(0, num_bins - 1)

    if edges.shape[1] != num_bins + 1:
        raise ValueError(f"Expected {num_bins + 1} bin edges, got {edges.shape[1]}")
    edges = _reshape_edges_for_actions(edges, actions)
    flat_actions = actions.reshape(-1, actions.shape[-1])
    bins_by_dim = []
    for dim_idx in range(flat_actions.shape[-1]):
        cur_edges = edges[dim_idx]
        cur_values = flat_actions[:, dim_idx].clamp(cur_edges[0], cur_edges[-1])
        cur_bins = torch.bucketize(cur_values, cur_edges[1:-1], right=False)
        bins_by_dim.append(cur_bins)
    return torch.stack(bins_by_dim, dim=-1).reshape(actions.shape).long().clamp_(0, num_bins - 1)


def undiscretize_action_bins(
    action_bins: torch.Tensor | Sequence[int],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
    bin_edges: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> torch.FloatTensor:
    action_bins = torch.as_tensor(action_bins, dtype=torch.long)
    edges = normalize_action_bin_edges(bin_edges, device=action_bins.device, dtype=torch.float32)
    if edges is None:
        action_bins_float = action_bins.to(torch.float32)
        scale = (action_max - action_min) / (num_bins - 1)
        return action_min + action_bins_float * scale

    if edges.shape[1] != num_bins + 1:
        raise ValueError(f"Expected {num_bins + 1} bin edges, got {edges.shape[1]}")
    edges = _reshape_edges_for_actions(edges, action_bins)
    flat_bins = action_bins.reshape(-1, action_bins.shape[-1]).clamp(0, num_bins - 1)
    values_by_dim = []
    for dim_idx in range(flat_bins.shape[-1]):
        cur_edges = edges[dim_idx]
        cur_bins = flat_bins[:, dim_idx]
        left = cur_edges.index_select(0, cur_bins)
        right = cur_edges.index_select(0, cur_bins + 1)
        values_by_dim.append((left + right) * 0.5)
    return torch.stack(values_by_dim, dim=-1).reshape(action_bins.shape).to(torch.float32)


def bins_to_token_ids(
    action_bins: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
) -> torch.LongTensor:
    action_bins = torch.as_tensor(action_bins, dtype=torch.long)
    token_tensor = torch.as_tensor(
        action_token_ids,
        dtype=torch.long,
        device=action_bins.device,
    )
    return token_tensor[action_bins]


def token_ids_to_bins(
    token_ids: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
) -> torch.LongTensor:
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_tensor = torch.as_tensor(
        action_token_ids,
        dtype=torch.long,
        device=token_ids.device,
    )

    positions = torch.searchsorted(token_tensor, token_ids)
    valid = positions < token_tensor.numel()
    valid &= token_tensor[positions.clamp_max(token_tensor.numel() - 1)] == token_ids
    if not torch.all(valid):
        invalid_ids = token_ids[~valid].unique().tolist()
        raise ValueError(f"Found non-action token ids: {invalid_ids}")
    return positions


def actions_to_token_ids(
    actions: torch.Tensor | Sequence[float],
    action_token_ids: Sequence[int],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
    bin_edges: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> torch.LongTensor:
    action_bins = discretize_actions(
        actions,
        num_bins=num_bins,
        action_min=action_min,
        action_max=action_max,
        bin_edges=bin_edges,
    )
    return bins_to_token_ids(action_bins, action_token_ids)


def token_ids_to_actions(
    token_ids: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
    bin_edges: torch.Tensor | Sequence[Sequence[float]] | None = None,
) -> torch.FloatTensor:
    action_bins = token_ids_to_bins(token_ids, action_token_ids)
    return undiscretize_action_bins(
        action_bins,
        num_bins=num_bins,
        action_min=action_min,
        action_max=action_max,
        bin_edges=bin_edges,
    )
