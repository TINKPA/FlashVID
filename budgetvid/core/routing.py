"""Phase B -- routing: one score, two cut points.

Pure tensor code, no repo-internal imports.

Spec: 2026-08-04_method_budget-aware-token-pruning_v1.1 §3.2.
"""

from __future__ import annotations

import math

import torch


def frame_percentile(v: torch.Tensor) -> torch.Tensor:
    """Rank percentile of each frame's value among all L frames, in [0, 1].

    Used for alpha_t, which is a CROSS-frame comparison and therefore must be fed
    the raw per-frame redundancy mean, never the rank-normalised one (whose frame
    mean is 0.5 for every frame by construction).

    With L == 1 there is no cross-frame information; percentile is defined as 0,
    which sends alpha_t to alpha_max.
    """
    L = v.shape[0]
    if L == 1:
        return torch.zeros_like(v)
    order = v.float().argsort(stable=True)
    ranks = torch.empty_like(order)
    ranks.scatter_(0, order, torch.arange(L, device=v.device))
    return ranks.float() / (L - 1)


def beta_window(beta: float, r: float) -> str:
    """Which side of the effective window this (beta, r) lands on.

    N_active/N_f ~= beta*r, so the merge pool reaches the S=0 diagonal -- the
    "important but redundant" cell the method is motivated by -- only when
    beta*r >= 0.5, and something is still dropped only while beta*r < 1.
    Recorded per run so a later reader can tell what condition a row was taken
    under (plan H4 / test 10).
    """
    br = beta * r
    return "in" if 0.5 <= br < 1.0 else ("below" if br < 0.5 else "above")


def route_tokens(
    S: torch.Tensor,
    R_raw_mean: torch.Tensor,
    b_t: torch.Tensor,
    alpha_min: float = 0.4,
    alpha_max: float = 0.8,
    beta: float = 4.0,
    force_alpha: float | None = None,
    alpha_flip: bool = False,
) -> dict:
    """Phase B.

    S: [L, N_f] routing score. R_raw_mean: [L]. b_t: [L] ints summing to B.

    Returns per-frame index tensors for the three paths plus the scalars that
    produced them. Indices are into the frame's own [N_f] axis and are returned
    in descending-S order; Phase D re-sorts by original index.

    force_alpha pins alpha_t (force_alpha=1.0 gives B_M=0 and an empty merge pool,
    which is what makes the Step 4 identity-degeneracy test possible at all).
    alpha_flip reverses the mapping direction for the Step 6 ablation.
    """
    L, N_f = S.shape
    if b_t.shape[0] != L:
        raise ValueError(f"b_t has {b_t.shape[0]} entries but S has {L} frames")
    if beta < 1:
        raise ValueError(f"beta must be >= 1, got {beta}")

    if force_alpha is not None:
        alpha = torch.full((L,), float(force_alpha), device=S.device)
    else:
        p = frame_percentile(R_raw_mean)
        alpha = (alpha_min + (alpha_max - alpha_min) * p if alpha_flip
                 else alpha_max - (alpha_max - alpha_min) * p)

    order = S.argsort(dim=-1, descending=True, stable=True)

    retain, pool, drop, B_R_l, B_M_l, N_act_l = [], [], [], [], [], []
    for t in range(L):
        bt = int(b_t[t])
        if bt < 1:
            raise ValueError(f"b_t[{t}]={bt}: an empty frame is not allowed (spec §3.2)")
        if bt > N_f:
            raise ValueError(f"b_t[{t}]={bt} > N_f={N_f}")
        B_R = int(math.floor(float(alpha[t]) * bt))
        B_M = bt - B_R
        N_act = min(N_f, math.ceil(beta * bt))
        # b_t <= N_f and beta >= 1  =>  N_act >= b_t  =>  |pool| = N_act - B_R >= B_M
        assert N_act >= bt, (N_act, bt)
        assert N_act - B_R >= B_M, (N_act, B_R, B_M)

        o = order[t]
        retain.append(o[:B_R])
        if B_M == 0:
            # Nothing to merge into, so the "pool" would be computed and then
            # discarded by Phase C, and those tokens would silently vanish. Put
            # them where they actually go -- drop -- so the returned partition
            # describes what happens. This is the alpha_t=1 pure-pruning
            # degeneracy (ablation row B) and the precondition for the Step 4
            # identity test.
            pool.append(o[:0])
            drop.append(o[B_R:])
        else:
            pool.append(o[B_R:N_act])
            drop.append(o[N_act:])
        B_R_l.append(B_R); B_M_l.append(B_M); N_act_l.append(N_act)

    return {
        "retain_idx": retain,
        "pool_idx": pool,
        "drop_idx": drop,
        "alpha": alpha,
        "B_R": torch.tensor(B_R_l),
        "B_M": torch.tensor(B_M_l),
        "N_active": torch.tensor(N_act_l),
    }
