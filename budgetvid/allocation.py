"""Budget allocation policies.

This is the research surface of BudgetVID. FlashVID spends the same per-frame
token budget on every segment of the video (``flashvid/utils.py:46``), so a
static shot and a busy one are compressed equally hard. A policy here decides,
given a global budget, how many tokens per frame each segment gets.

A policy is a callable::

    policy(
        video_features,    # (num_frames, num_visual_tokens, feat_dim)
        cls_attention,     # (num_frames, num_visual_tokens)
        segment_lengths,   # List[int], sums to num_frames
        num_visual_tokens, # int, tokens per frame before compression
        config,            # BudgetVidConfig
    ) -> List[int]         # per-frame budget for each segment

It returns one integer per segment: the token budget *per frame* inside that
segment. Segment ``i`` therefore consumes ``budgets[i] * segment_lengths[i]``
tokens in total, and the sum over segments must not exceed
``global_budget(...)``, which is the exact number of tokens FlashVID would have
spent at the same ``retention_ratio``. That constraint is what keeps a
comparison against fixed-ratio baselines honest, and `validate` enforces it.

Policies must be deterministic: two runs on the same video have to allocate the
same budget, otherwise benchmark numbers stop being reproducible.
"""

import math
from typing import Callable, Dict, List

_ALLOCATION_REGISTRY: Dict[str, Callable] = {}


def register_allocation(name: str, fn: Callable) -> None:
    """Register an allocation policy under ``name``."""
    _ALLOCATION_REGISTRY[name] = fn


def available_allocations() -> List[str]:
    """Names of every registered policy, for error messages and logging."""
    return sorted(_ALLOCATION_REGISTRY)


def per_frame_budget(num_visual_tokens: int, config) -> int:
    """The per-frame token budget FlashVID would use, unchanged.

    This is ``flashvid/utils.py:46`` verbatim. Kept here so every policy shares
    one definition of "what one frame's fair share is".

    Args:
        num_visual_tokens (int): Tokens per frame before compression.
        config (BudgetVidConfig): Supplies ``retention_ratio`` and ``expansion``.

    Returns:
        int: Token budget for a single frame.
    """
    return math.ceil(num_visual_tokens * config.retention_ratio * config.expansion)


def global_budget(num_frames: int, num_visual_tokens: int, config) -> int:
    """Total tokens the whole video may spend across all segments.

    Args:
        num_frames (int): Number of frames in the video.
        num_visual_tokens (int): Tokens per frame before compression.
        config (BudgetVidConfig): Supplies ``retention_ratio`` and ``expansion``.

    Returns:
        int: The global token budget.
    """
    return num_frames * per_frame_budget(num_visual_tokens, config)


def spend(budgets: List[int], segment_lengths: List[int]) -> int:
    """Tokens consumed by an allocation.

    Args:
        budgets (List[int]): Per-frame budget for each segment.
        segment_lengths (List[int]): Frame count of each segment.

    Returns:
        int: Total tokens the allocation spends.
    """
    return sum(int(b) * int(n) for b, n in zip(budgets, segment_lengths))


def validate(budgets: List[int], segment_lengths: List[int], num_frames: int, num_visual_tokens: int, config) -> None:
    """Check an allocation is well formed and within budget.

    Args:
        budgets (List[int]): Per-frame budget for each segment.
        segment_lengths (List[int]): Frame count of each segment.
        num_frames (int): Number of frames in the video.
        num_visual_tokens (int): Tokens per frame before compression.
        config (BudgetVidConfig): Supplies the budget and ``enforce_budget``.

    Raises:
        ValueError: If the shape is wrong, a budget is not a positive int, or
            the allocation overspends while ``config.enforce_budget`` is set.
    """
    if len(budgets) != len(segment_lengths):
        raise ValueError(f"Allocation returned {len(budgets)} budgets for {len(segment_lengths)} segments.")
    for i, b in enumerate(budgets):
        if int(b) != b or b < 1:
            raise ValueError(f"Budget for segment {i} must be a positive integer, got {b!r}.")
        if b > num_visual_tokens:
            raise ValueError(f"Budget for segment {i} is {b}, above the {num_visual_tokens} tokens a frame actually has.")

    allowed = global_budget(num_frames, num_visual_tokens, config)
    used = spend(budgets, segment_lengths)
    if used > allowed and config.enforce_budget:
        raise ValueError(f"Allocation '{config.allocation}' spends {used} tokens but the budget is {allowed} " f"(retention_ratio={config.retention_ratio}, expansion={config.expansion}). " f"Comparisons against fixed-ratio baselines are only valid at or below the budget. " f"Set enforce_budget=False to override deliberately.")


def allocate(video_features, cls_attention, segment_lengths: List[int], num_visual_tokens: int, config) -> List[int]:
    """Run the configured allocation policy and validate its output.

    Args:
        video_features (torch.Tensor): (num_frames, num_visual_tokens, feat_dim).
        cls_attention (torch.Tensor): (num_frames, num_visual_tokens).
        segment_lengths (List[int]): Frame count of each segment.
        num_visual_tokens (int): Tokens per frame before compression.
        config (BudgetVidConfig): Supplies ``allocation`` and the budget.

    Returns:
        List[int]: Per-frame token budget for each segment.
    """
    name = config.allocation
    if name not in _ALLOCATION_REGISTRY:
        raise KeyError(f"Unknown allocation policy '{name}'. Registered: {available_allocations()}.")

    budgets = _ALLOCATION_REGISTRY[name](
        video_features=video_features,
        cls_attention=cls_attention,
        segment_lengths=segment_lengths,
        num_visual_tokens=num_visual_tokens,
        config=config,
    )
    budgets = [int(b) for b in budgets]
    num_frames = sum(segment_lengths)
    validate(budgets, segment_lengths, num_frames, num_visual_tokens, config)
    return budgets


def uniform_allocation(segment_lengths: List[int], num_visual_tokens: int, config, **_) -> List[int]:
    """Give every segment the same per-frame budget.

    This reproduces FlashVID exactly, which makes it the regression baseline:
    ``allocation=uniform`` must match ``enable_flashvid=True`` to the decimal on
    every benchmark. If it ever does not, the plumbing is broken, not the idea.

    Returns:
        List[int]: The same per-frame budget repeated once per segment.
    """
    return [per_frame_budget(num_visual_tokens, config)] * len(segment_lengths)


register_allocation("uniform", uniform_allocation)
