"""Where the delivered visual tokens sit in the LLM's sequence.

Every consumer of the visual span (the log-mass bias, the FastV keep/mass
bookkeeping, the inner-LLM dump records) used to assume the visual tokens form
one contiguous slice ``[visual_token_start_index, + visual_token_length)``.
That holds on Qwen2.5-VL and LLaVA-OneVision. It does NOT hold on Qwen3-VL:
``flashvid/modeling_qwen3_vl.py`` keeps every non-visual token of the video
prompt, so after compression non-visual tokens sit between the visual ones
(2026-10-15 smoke: 219 positions survived FastV inside the span, 144 of them
visual).

The fix: the Qwen3 forward records the true positions on the config as
``visual_token_positions`` (1-D long tensor, entry i = sequence position of the
i-th delivered visual token, in the same order as ``token_mass``). Everything
here falls back to the contiguous span when that attribute is absent, so the
other backbones are byte-for-byte unchanged.
"""

from __future__ import annotations

import torch


def positions_of(cfg, n_vis: int | None = None, device=None) -> torch.Tensor:
    """Sequence positions of the visual tokens, in token_mass order."""
    pos = getattr(cfg, "visual_token_positions", None)
    if pos is not None:
        pos = pos.to(device) if device is not None else pos
        if n_vis is not None and pos.numel() != n_vis:
            raise AssertionError(f"visual_token_positions has {pos.numel()} entries, expected {n_vis}")
        return pos
    start = int(getattr(cfg, "visual_token_start_index", 0))
    n = int(getattr(cfg, "visual_token_length", 0)) if n_vis is None else int(n_vis)
    return torch.arange(start, start + n, device=device)


def kept_ranks(cfg, keep_indices: torch.Tensor, n_vis: int) -> torch.Tensor:
    """Ranks (into the visual list) of the visual tokens that survive ``keep_indices``."""
    pos = positions_of(cfg, n_vis, device=keep_indices.device)
    return torch.nonzero(torch.isin(pos, keep_indices), as_tuple=False).flatten()


def remap_after_keep(cfg, keep_indices: torch.Tensor, ranks: torch.Tensor) -> None:
    """After a sequence-level keep, point ``visual_token_positions`` at the new slots.

    No-op on the contiguous backbones (no attribute set): their span stays at
    the same start and shrinks, which ``visual_token_length`` already says.
    """
    pos = getattr(cfg, "visual_token_positions", None)
    if pos is None:
        return
    old = pos.to(keep_indices.device)[ranks]
    cfg.visual_token_positions = torch.searchsorted(keep_indices, old)


def clear(cfg) -> None:
    cfg.visual_token_positions = None
