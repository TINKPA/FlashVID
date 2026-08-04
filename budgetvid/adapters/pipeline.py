"""Registry entry point: one policy switch, one assembly path.

Registered under the method name ``bv``. Which policy runs is
``config.policy``; everything after the policy is shared, which is what makes an
ablation row comparable to a baseline row rather than a different program that
happens to produce a similar number.

Budget is an ABSOLUTE token count, per the spec: B = round(r * N) with N = L*N_f,
split across frames by largest remainder. Note this differs from FlashVID's own
`retention_ratio`, which is a per-LLM-layer average and lands ~30% higher in
actual visual tokens (see experiments/flashvid_token_accounting). Do not read the
two as the same knob.
"""

from __future__ import annotations

import torch

from ..core.assembly import assemble
from ..core.budget import split_budget
from ..core.policies import random_drop_keep, uniform_keep

POLICIES = ("none", "random_drop", "uniform")


def budgetvid_pipeline(video_features: torch.Tensor, cls_attention: torch.Tensor,
                       flashvid_config) -> tuple[torch.Tensor, torch.Tensor]:
    """Vision-side compression for method ``bv``.

    Args:
        video_features: [L, N_f, D] -- post-projector, post-pool (H3 decision).
        cls_attention: [L, N_f] -- importance, see spec §3.1.

    Returns:
        ``(tokens, global_indices)`` sorted by global index, matching the
        contract of ``flashvid.utils.flashvid_compression``.
    """
    L, N_f, _ = video_features.shape
    policy = getattr(flashvid_config, "policy", "none")
    device = video_features.device

    if policy == "none":
        g = torch.arange(L * N_f, dtype=torch.long, device=device)
        flashvid_config.visual_token_length = L * N_f
        return video_features.reshape(L * N_f, -1), g

    if policy not in POLICIES:
        raise KeyError(f"unknown policy '{policy}'; known: {sorted(POLICIES)}")

    r = float(flashvid_config.retention_ratio)
    B = int(round(r * L * N_f))
    b_t = split_budget(B, L, N_f).to(device)

    seed = int(getattr(flashvid_config, "seed", 42))
    if policy == "uniform":
        keep = uniform_keep(L, N_f, b_t, device=device)
    else:
        keep = random_drop_keep(L, N_f, b_t, seed=seed, device=device)

    for t, k in enumerate(keep):
        assert k.numel() == int(b_t[t]), (t, k.numel(), int(b_t[t]))

    tokens, g = assemble(video_features, keep, merged=None, expected_total=B)
    flashvid_config.visual_token_length = int(tokens.shape[0])
    return tokens, g
