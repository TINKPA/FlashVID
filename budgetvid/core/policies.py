"""The trivial floor policies (plan Step 2.5).

random-drop and uniform-downsample are sanity checks, not results: any method
that fails to beat them has a bug rather than a tuning problem. They are cheap,
and they exercise the same budget arithmetic and the same assembly path as the
real method, so running them first catches budget-off-by-one, ordering and
duplicate-index errors before a $3 evaluation depends on them being right.

Each returns per-frame within-frame indices to keep, exactly ``b_t`` of them.
"""

from __future__ import annotations

import torch


def uniform_keep(L: int, N_f: int, b_t: torch.Tensor, device=None) -> list[torch.Tensor]:
    """Evenly spaced grid positions -- a fixed spatial subsample, same every frame
    of equal budget. Deterministic by construction."""
    out = []
    for t in range(L):
        k = int(b_t[t])
        idx = torch.linspace(0, N_f - 1, k, device=device).round().long()
        # linspace+round can collide when k is close to N_f; dedupe and top up so
        # the frame still spends exactly its budget.
        idx = torch.unique(idx)
        if idx.numel() < k:
            pool = torch.ones(N_f, dtype=torch.bool, device=device)
            pool[idx] = False
            extra = torch.nonzero(pool, as_tuple=False).flatten()[: k - idx.numel()]
            idx = torch.cat([idx, extra]).sort().values
        out.append(idx[:k])
    return out


def random_drop_keep(L: int, N_f: int, b_t: torch.Tensor, seed: int = 42,
                     device=None) -> list[torch.Tensor]:
    """Uniformly random positions per frame.

    Seeded off a private generator rather than global RNG state: benchmark numbers
    have to reproduce, and the eval harness seeds torch for its own reasons.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    out = []
    for t in range(L):
        k = int(b_t[t])
        perm = torch.randperm(N_f, generator=g)[:k].sort().values
        out.append(perm.to(device) if device is not None else perm)
    return out
