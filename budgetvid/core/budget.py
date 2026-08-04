"""Per-frame budget allocation.

Pure tensor/int code with no repo-internal imports, so it unit-tests on a laptop
with no CUDA and no decord.

Spec: 2026-08-04_method_budget-aware-token-pruning_v1.1 §2.2.
"""

from __future__ import annotations

import torch


def split_budget(B: int, L: int, N_f: int | None = None) -> torch.Tensor:
    """Split a total token budget B across L frames, largest-remainder style.

    v1 of the spec wrote b_t = B/L, which is incompatible with b_t integral and
    sum(b_t) == B whenever L does not divide B -- and 627/32 does not. The first
    ``B % L`` frames get the extra token:

        b_t = B//L + (1 if t < B % L else 0)      for t = 0..L-1

    Returns int64 tensor [L]. Raises if the result cannot satisfy the invariants,
    rather than silently returning a budget the router cannot honour.
    """
    if L <= 0:
        raise ValueError(f"L must be positive, got {L}")
    if B < L:
        raise ValueError(f"B={B} < L={L}: some frame would get b_t=0, which means the "
                         "frame disappears entirely; not allowed (spec §3.2)")
    base, rem = divmod(int(B), int(L))
    b = torch.full((L,), base, dtype=torch.long)
    if rem:
        b[:rem] += 1
    if N_f is not None and int(b.max()) > N_f:
        raise ValueError(f"b_t={int(b.max())} exceeds N_f={N_f}: budget larger than the frame")
    assert int(b.sum()) == int(B), (int(b.sum()), B)
    return b
