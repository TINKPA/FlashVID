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


def attn_top_keep(I: torch.Tensor, b_t: torch.Tensor,
                  sink_cells: torch.Tensor | None = None) -> list[torch.Tensor]:
    """Per-frame top-``b_t`` positions by raw importance (Round-2 B+ probe).

    The retain-only diagnostic that trusts the attention signal completely --
    equivalent to ``prune_only`` with ``lam=0`` up to tie-breaking (topk on the
    raw signal vs. stable-argsort ranks). With ``sink_cells`` given, those grid
    positions are excluded BEFORE ranking (``attn_top_nosink``); the difference
    between the two rows is the net effect of sink tokens on a pure-attention
    selector at that budget, pre-registered to grow as the budget tightens.
    """
    L, N_f = I.shape
    I = I.detach().float().clone()
    n_sink = 0
    if sink_cells is not None and sink_cells.numel():
        sink_cells = sink_cells.to(I.device).long()
        if int(sink_cells.max()) >= N_f or int(sink_cells.min()) < 0:
            raise ValueError(f"sink cell index out of range for N_f={N_f}")
        I[:, sink_cells] = float("-inf")
        n_sink = int(sink_cells.numel())
    out = []
    for t in range(L):
        k = int(b_t[t])
        if k > N_f - n_sink:
            raise ValueError(
                f"frame {t}: budget {k} exceeds the {N_f - n_sink} non-sink cells")
        idx = I[t].topk(k).indices.sort().values
        out.append(idx)
    return out
