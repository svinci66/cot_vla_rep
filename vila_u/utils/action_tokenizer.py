from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch

from vila_u.constants import ACTION_MAX, ACTION_MIN, ACTION_NUM_BINS


@dataclass(frozen=True)
class ActionTokenSpec:
    token_ids: tuple[int, ...]
    num_bins: int = ACTION_NUM_BINS
    action_min: float = ACTION_MIN
    action_max: float = ACTION_MAX

    def __post_init__(self) -> None:
        if len(self.token_ids) != self.num_bins:
            raise ValueError(
                f"Expected {self.num_bins} action tokens, got {len(self.token_ids)}"
            )


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


def discretize_actions(
    actions: torch.Tensor | Sequence[float],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
) -> torch.LongTensor:
    actions = torch.as_tensor(actions, dtype=torch.float32)
    clipped = actions.clamp(action_min, action_max)
    scale = (num_bins - 1) / (action_max - action_min)
    bins = torch.round((clipped - action_min) * scale).to(torch.long)
    return bins.clamp_(0, num_bins - 1)


def undiscretize_action_bins(
    action_bins: torch.Tensor | Sequence[int],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
) -> torch.FloatTensor:
    action_bins = torch.as_tensor(action_bins, dtype=torch.float32)
    scale = (action_max - action_min) / (num_bins - 1)
    return action_min + action_bins * scale


def bins_to_token_ids(
    action_bins: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
) -> torch.LongTensor:
    action_bins = torch.as_tensor(action_bins, dtype=torch.long)
    token_tensor = torch.as_tensor(action_token_ids, dtype=torch.long)
    return token_tensor[action_bins]


def token_ids_to_bins(
    token_ids: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
) -> torch.LongTensor:
    token_ids = torch.as_tensor(token_ids, dtype=torch.long)
    token_tensor = torch.as_tensor(action_token_ids, dtype=torch.long)

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
) -> torch.LongTensor:
    action_bins = discretize_actions(
        actions,
        num_bins=num_bins,
        action_min=action_min,
        action_max=action_max,
    )
    return bins_to_token_ids(action_bins, action_token_ids)


def token_ids_to_actions(
    token_ids: torch.Tensor | Sequence[int],
    action_token_ids: Sequence[int],
    num_bins: int = ACTION_NUM_BINS,
    action_min: float = ACTION_MIN,
    action_max: float = ACTION_MAX,
) -> torch.FloatTensor:
    action_bins = token_ids_to_bins(token_ids, action_token_ids)
    return undiscretize_action_bins(
        action_bins,
        num_bins=num_bins,
        action_min=action_min,
        action_max=action_max,
    )
