"""Phase D -- assemble the surviving tokens back into a sequence.

Every policy shares this function. That is not tidiness: the plan makes the
comparability of ablations against baselines depend on all of them travelling the
same assembly and position-id path, so a second copy of this logic would quietly
invalidate the comparison it exists to support.

Spec: 2026-08-04_method_budget-aware-token-pruning_v1.2 §2.2, §3.5.

Contract, inherited from `flashvid.utils.flashvid_compression` so the registry
call sites need no special-casing:

    returns (tokens [B', D], global_indices [B']) sorted by global index ascending

where the global index of frame ``t``, grid position ``i`` is ``t * N_f + i``.

On this backbone (LLaVA-OneVision -> Qwen2, 1-D RoPE) that ordering IS the
position information -- `flashvid/llava_arch.py` discards the returned indices and
the LLM assigns positions by sequence order. §3.5's warning about M-RoPE time axes
applies to Qwen2/2.5-VL, not here, which is exactly why the spec says pi degenerates
to a scalar on 1-D RoPE backbones. Sorting correctly is therefore necessary AND
sufficient, and getting it wrong fails silently -- hence the asserts below.
"""

from __future__ import annotations

import torch


def assemble(
    video_features: torch.Tensor,
    retain_idx: list[torch.Tensor],
    merged: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    expected_total: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the compressed sequence.

    Args:
        video_features: [L, N_f, D].
        retain_idx: per frame, within-frame indices kept verbatim.
        merged: per frame, ``(features [k, D], seed_idx [k])``. Merged tokens
            inherit their seed's position (spec §2.2), never a coordinate average.
        expected_total: if given, assert the output has exactly this many tokens.

    Returns:
        ``(tokens, global_indices)`` sorted by global index.
    """
    L, N_f, D = video_features.shape
    if len(retain_idx) != L:
        raise ValueError(f"retain_idx has {len(retain_idx)} frames, features have {L}")
    if merged is not None and len(merged) != L:
        raise ValueError(f"merged has {len(merged)} frames, features have {L}")

    feats, gidx = [], []
    for t in range(L):
        keep = retain_idx[t]
        if keep.numel():
            if int(keep.max()) >= N_f or int(keep.min()) < 0:
                raise ValueError(f"frame {t}: retain index out of range [0,{N_f})")
            feats.append(video_features[t, keep])
            gidx.append(t * N_f + keep.to(torch.long))
        if merged is not None:
            mfeat, mseed = merged[t]
            if mfeat.numel():
                if mfeat.shape[0] != mseed.shape[0]:
                    raise ValueError(f"frame {t}: {mfeat.shape[0]} merged tokens vs {mseed.shape[0]} seeds")
                feats.append(mfeat.to(video_features.dtype))
                gidx.append(t * N_f + mseed.to(torch.long))

    if not feats:
        raise ValueError("assemble produced no tokens; every frame was emptied")

    tokens = torch.cat(feats, 0)
    g = torch.cat(gidx, 0)

    # A duplicated global index means two output tokens claim one input position:
    # the sequence would still have the right length and the model would still
    # run, so nothing would surface except a slow accuracy loss.
    if torch.unique(g).numel() != g.numel():
        raise AssertionError("duplicate global indices -- a token was emitted twice "
                             "(retain and merge sets are not disjoint?)")
    if expected_total is not None and tokens.shape[0] != expected_total:
        raise AssertionError(f"assembled {tokens.shape[0]} tokens, budget said {expected_total}")

    order = g.argsort()
    return tokens[order], g[order]
