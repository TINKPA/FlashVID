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
    masses: list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the compressed sequence.

    Args:
        video_features: [L, N_f, D].
        retain_idx: per frame, within-frame indices kept verbatim.
        merged: per frame, ``(features [k, D], seed_idx [k])``. Merged tokens
            inherit their seed's position (spec §2.2), never a coordinate average.
        expected_total: if given, assert the output has exactly this many tokens.
        masses: per frame, the group size of each merged token (BudgetVID 2.0).
            When given, the return gains a third element carrying those masses
            in the SAME order as the tokens -- they have to travel through this
            one sort with the tokens, or the log-mass bias lands on the wrong
            keys, which is a silent accuracy loss rather than an error.

    Returns:
        ``(tokens, global_indices)``, or ``(tokens, global_indices, mass)`` when
        ``masses`` is given, sorted by global index.
    """
    L, N_f, D = video_features.shape
    if len(retain_idx) != L:
        raise ValueError(f"retain_idx has {len(retain_idx)} frames, features have {L}")
    if merged is not None and len(merged) != L:
        raise ValueError(f"merged has {len(merged)} frames, features have {L}")

    feats, gidx, mass = [], [], []
    for t in range(L):
        keep = retain_idx[t]
        if keep.numel():
            if int(keep.max()) >= N_f or int(keep.min()) < 0:
                raise ValueError(f"frame {t}: retain index out of range [0,{N_f})")
            feats.append(video_features[t, keep])
            gidx.append(t * N_f + keep.to(torch.long))
            if masses is not None:
                # a retained token is a group of one, not a special case
                mass.append(torch.ones(keep.numel(), device=keep.device))
        if merged is not None:
            mfeat, mseed = merged[t]
            if mfeat.numel():
                if mfeat.shape[0] != mseed.shape[0]:
                    raise ValueError(f"frame {t}: {mfeat.shape[0]} merged tokens vs {mseed.shape[0]} seeds")
                feats.append(mfeat.to(video_features.dtype))
                gidx.append(t * N_f + mseed.to(torch.long))
                if masses is not None:
                    if masses[t].shape[0] != mfeat.shape[0]:
                        raise ValueError(f"frame {t}: {mfeat.shape[0]} merged tokens vs {masses[t].shape[0]} masses")
                    mass.append(masses[t].float())

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
    if masses is not None:
        return tokens[order], g[order], torch.cat(mass, 0)[order]
    return tokens[order], g[order]


def reconstruct_full(tokens: torch.Tensor, gidx: torch.Tensor,
                     group_of: torch.Tensor) -> torch.Tensor:
    """Expand the compressed set back to full length: every original position
    gets its group's delivered token.

    The inverse direction of ``assemble``: where assemble throws away the
    members and keeps one token per group, this hands every member position a
    copy of that token. Positions are exact by construction -- the caller emits
    the result with ``global_indices = arange(L * N_f)``, so the backbone's
    position-id filtering degenerates to the identity and every copy sits at
    its member's ORIGINAL position (M-RoPE included). What stays lost is only
    the within-group feature residual.

    Args:
        tokens: [B, D] delivered tokens, sorted by global index (assemble's
            output order).
        gidx: [B] their global indices, ascending (assemble's output order).
        group_of: [L, N_f] for every original token, the global index of the
            token its group was delivered as (``compress_video``'s map). Every
            entry must occur in ``gidx``.

    Returns:
        [L * N_f, D] reconstruction, same dtype as ``tokens``.
    """
    flat = group_of.reshape(-1).to(gidx.device)
    row = torch.searchsorted(gidx, flat)
    # searchsorted returns an insertion point, which only equals the match when
    # the value is actually present; a group_of entry pointing at a token that
    # was never delivered would silently grab its neighbour's feature.
    if int(row.max()) >= gidx.numel() or not torch.equal(gidx[row], flat):
        raise AssertionError("group_of names a global index that assemble never delivered")
    return tokens[row]


def reconstruct_rank(tokens: torch.Tensor, gidx: torch.Tensor,
                     group_of: torch.Tensor, x: torch.Tensor,
                     rank: int) -> torch.Tensor:
    """Full-length reconstruction with a rank-``rank`` residual code per group.

    The mass-as-vector ladder: each group additionally records the top-``rank``
    principal directions of its members' residuals against the delivered token
    (the "angles" of the member set) and one coefficient per member per
    direction. Reconstruction is ``x̂_i = delivered_j + sum_t a_{i,t} u_{j,t}``.

    rank=0 is exactly ``reconstruct_full`` (count + positions only). As rank
    reaches each group's member count the residual matrix is represented
    exactly and the reconstruction converges to the original tokens, so the
    two ends of the ladder coincide with the group / identity oracles.

    Args:
        tokens: [B, D] delivered tokens, sorted by global index.
        gidx: [B] their global indices, ascending.
        group_of: [L, N_f] member -> delivered-token global index.
        x: [L, N_f, D] the ORIGINAL tokens (delivery space).
        rank: residual directions kept per group; 0 = none.

    Returns:
        [L * N_f, D] reconstruction, float32.
    """
    full = reconstruct_full(tokens, gidx, group_of).float().clone()
    if rank <= 0:
        return full
    flat_g = group_of.reshape(-1)
    xf = x.reshape(-1, x.shape[-1]).float()
    resid = xf - full
    for g in gidx.tolist():
        mem = torch.nonzero(flat_g == g).flatten()
        if mem.numel() < 2:
            # a singleton's residual is 0 by construction (identity delivery)
            # or a single vector: rank-1 represents it exactly either way.
            full[mem] = xf[mem]
            continue
        R = resid[mem]                                    # [m, D]
        r = min(rank, R.shape[0])
        # Top-r right singular vectors of the residual matrix; lowrank SVD has
        # RNG inside, and determinism is load-bearing here, so use full SVD.
        _, _, Vh = torch.linalg.svd(R, full_matrices=False)
        U = Vh[:r]                                        # [r, D] directions
        full[mem] = full[mem] + (R @ U.t()) @ U           # project + add back
    return full
