"""Phase C -- seeded merge (spec §3.3 variant a).

The merge pool's top-B_M tokens by S become seeds; every other pool token is
assigned to its nearest seed by cosine similarity and each seed emits one
size-weighted mean. One shot, exactly B_M outputs, fully vectorised.

Merged tokens inherit their SEED's position, never a coordinate average -- an
averaged coordinate can land outside the object it came from (spec §2.2, §3.5).
"""

from __future__ import annotations

import torch


def seeded_merge(
    X_frame: torch.Tensor,
    pool_idx: torch.Tensor,
    S_frame: torch.Tensor,
    B_M: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge one frame's pool into ``B_M`` tokens.

    Args:
        X_frame: [N_f, D] features for this frame.
        pool_idx: within-frame indices of the merge pool, in descending-S order.
        S_frame: [N_f] routing scores.
        B_M: how many tokens to emit.

    Returns:
        ``(merged [B_M, D], seed_idx [B_M])`` -- seed_idx are within-frame
        indices, used by Phase D for position inheritance.
    """
    if B_M <= 0 or pool_idx.numel() == 0:
        empty = X_frame.new_zeros((0, X_frame.shape[-1]))
        return empty, pool_idx.new_zeros((0,))
    if pool_idx.numel() < B_M:
        raise ValueError(f"pool has {pool_idx.numel()} tokens but B_M={B_M}; "
                         "the routing invariant N_active - B_R >= B_M was violated")

    # pool_idx arrives sorted by descending S, so the first B_M are the seeds.
    # Sorting again here would be redundant but makes the function safe to call
    # with an arbitrary pool order.
    order = torch.argsort(S_frame[pool_idx], descending=True, stable=True)
    pool = pool_idx[order]
    seeds, rest = pool[:B_M], pool[B_M:]

    feats = X_frame[pool].float()
    seed_feats = X_frame[seeds].float()

    if rest.numel() == 0:
        # Nothing to absorb: each seed is its own singleton mean, i.e. itself.
        return X_frame[seeds], seeds

    sn = seed_feats / seed_feats.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    rf = X_frame[rest].float()
    rn = rf / rf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    assign = (rn @ sn.T).argmax(dim=-1)                       # [len(rest)]

    # size-weighted mean == plain mean over {seed} u {members assigned to it}
    sums = seed_feats.clone()
    sums.index_add_(0, assign, rf)
    counts = torch.ones(B_M, device=X_frame.device, dtype=sums.dtype)
    counts.index_add_(0, assign, torch.ones_like(assign, dtype=sums.dtype))
    merged = (sums / counts.unsqueeze(-1)).to(X_frame.dtype)
    return merged, seeds
