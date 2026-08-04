"""Phase A -- scoring: redundancy, rank normalisation, the routing score S.

Pure tensor code, no repo-internal imports, so it unit-tests without CUDA.

Spec: 2026-08-04_method_budget-aware-token-pruning_v1.1 §3.1.

Importance I is NOT computed here. It comes from the backbone (see the Step 1
report: on LLaVA-OneVision it is the last SigLIP layer's attention averaged over
heads and then over queries) and is passed in. Plan revision R14.
"""

from __future__ import annotations

import torch


def _l2norm(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def spatial_redundancy(X: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
    """Mean cosine similarity to the 4-neighbourhood.

    X: [L, N_f, D] with N_f == grid[0]*grid[1], row-major.
    Returns [L, N_f].

    Border tokens have fewer than 4 neighbours; the mean is over the neighbours
    that exist, not over 4 with zeros, otherwise the frame edge would look
    systematically less redundant than the interior for a purely geometric reason.
    """
    H, W = grid
    L, N_f, _ = X.shape
    if H * W != N_f:
        raise ValueError(f"grid {grid} does not match N_f={N_f}")

    Xn = _l2norm(X.float()).view(L, H, W, -1)
    total = torch.zeros(L, H, W, device=X.device, dtype=torch.float32)
    count = torch.zeros(L, H, W, device=X.device, dtype=torch.float32)

    # up / down / left / right, each as a shifted slice product
    def add(a_slice, b_slice, dst_slice):
        sim = (Xn[a_slice] * Xn[b_slice]).sum(-1)
        total[dst_slice] += sim
        count[dst_slice] += 1

    s = slice(None)
    add((s, slice(1, None), s), (s, slice(0, -1), s), (s, slice(1, None), s))   # up
    add((s, slice(0, -1), s), (s, slice(1, None), s), (s, slice(0, -1), s))     # down
    add((s, s, slice(1, None)), (s, s, slice(0, -1)), (s, s, slice(1, None)))   # left
    add((s, s, slice(0, -1)), (s, s, slice(1, None)), (s, s, slice(0, -1)))     # right

    return (total / count.clamp_min(1)).view(L, N_f)


def temporal_redundancy(X: torch.Tensor) -> torch.Tensor:
    """Cosine similarity to the same spatial position in the previous frame.

    X: [L, N_f, D]. Frame 0 is set to 0 per spec §3.1.

    Note the asymmetry this creates: frame 0's R is (1-eta)*R_sp only, so it is
    systematically lower than other frames' -- quantified as bias D8 in Step 2.
    """
    Xn = _l2norm(X.float())
    out = torch.zeros(X.shape[:2], device=X.device, dtype=torch.float32)
    if X.shape[0] > 1:
        out[1:] = (Xn[1:] * Xn[:-1]).sum(-1)
    return out


def rank_normalize(v: torch.Tensor) -> torch.Tensor:
    """Within-frame rank percentile in [0, 1]. v: [L, N_f] -> [L, N_f].

    Ordinal ranks via a stable double-argsort: deterministic, and ties break by
    index rather than by whatever order the sort happens to produce.

    Spec §3.1 divides by max(N_f-1, 1) so a degenerate N_f=1 frame does not
    divide by zero.
    """
    N_f = v.shape[-1]
    order = v.float().argsort(dim=-1, stable=True)
    ranks = torch.empty_like(order)
    ar = torch.arange(N_f, device=v.device).expand_as(order)
    ranks.scatter_(-1, order, ar)
    return ranks.float() / max(N_f - 1, 1)


def score_tokens(
    X: torch.Tensor,
    I: torch.Tensor,
    grid: tuple[int, int],
    eta: float = 0.5,
    lam: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Phase A.

    X: [L, N_f, D] features in the declared feature space.
    I: [L, N_f] importance, supplied by the backbone.

    Returns R_sp, R_tp, R_raw, R_raw_mean [L], I_hat, R_hat, S -- all the
    intermediates, because Step 2 has to plot them and a function that returned
    only S would force a second, possibly divergent, implementation.

    R_raw_mean is the per-frame mean of the RAW redundancy. Cross-frame
    comparison (alpha_t) must use it and never the rank-normalised R_hat, whose
    per-frame mean is 0.5 by construction.
    """
    if X.shape[:2] != I.shape:
        raise ValueError(f"X {tuple(X.shape)} and I {tuple(I.shape)} disagree on [L, N_f]")
    if not 0.0 <= eta <= 1.0:
        raise ValueError(f"eta must be in [0,1], got {eta}")

    R_sp = spatial_redundancy(X, grid)
    R_tp = temporal_redundancy(X)
    R_raw = (1.0 - eta) * R_sp + eta * R_tp

    I_hat = rank_normalize(I)
    R_hat = rank_normalize(R_raw)
    return {
        "R_sp": R_sp,
        "R_tp": R_tp,
        "R_raw": R_raw,
        "R_raw_mean": R_raw.mean(dim=-1),
        "I_hat": I_hat,
        "R_hat": R_hat,
        "S": I_hat - lam * R_hat,
    }
