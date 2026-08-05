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

import pathlib

import torch

from ..core.assembly import assemble
from ..core.budget import split_budget
from ..core.merging import seeded_merge
from ..core.policies import random_drop_keep, uniform_keep
from ..core.routing import route_tokens
from ..core.scoring import score_tokens

POLICIES = ("none", "random_drop", "uniform", "threeway", "prune_only", "merge_only")


def _rank01(v: torch.Tensor) -> torch.Tensor:
    """Within-frame rank percentile, matching core.scoring.rank_normalize."""
    n = v.shape[-1]
    o = v.argsort(dim=-1, stable=True)
    r = torch.empty_like(o)
    r.scatter_(-1, o, torch.arange(n, device=v.device).expand_as(o))
    return r.float() / max(n - 1, 1)


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
    if policy in ("uniform", "random_drop"):
        keep = (uniform_keep(L, N_f, b_t, device=device) if policy == "uniform"
                else random_drop_keep(L, N_f, b_t, seed=seed, device=device))
        for t, k in enumerate(keep):
            assert k.numel() == int(b_t[t]), (t, k.numel(), int(b_t[t]))
        tokens, g = assemble(video_features, keep, merged=None, expected_total=B)
        flashvid_config.visual_token_length = int(tokens.shape[0])
        return tokens, g

    # ---- the method itself: score -> route -> merge -> assemble ----
    cfg = flashvid_config
    grid = int(round(N_f ** 0.5))
    if grid * grid != N_f:
        raise ValueError(f"N_f={N_f} is not a square grid; the 4-neighbourhood needs one")

    I_used = cls_attention.float()
    if bool(getattr(cfg, "debias_pos", False)):
        # The importance signal is ~27% positional (Step 2): eight of the 196
        # cells sit in the frame's top-b_t 98.5% of the time. At b_t=2 that means
        # both retained tokens are the same two grid cells in every frame, so a
        # 256-frame video contributes 256 copies of one position instead of
        # coverage. Subtracting the position mean removes exactly that component
        # and leaves the content-dependent part.
        import numpy as _np
        _b = _np.load(pathlib.Path(__file__).resolve().parents[1] / "assets" / f"pos_baseline_pooled{N_f}.npy")
        base = torch.from_numpy(_b).to(I_used.device, I_used.dtype)
        I_used = _rank01(I_used) - base.unsqueeze(0)

    sc = score_tokens(video_features.float(), I_used, (grid, grid),
                      eta=float(getattr(cfg, "eta", 0.5)),
                      lam=float(getattr(cfg, "lam", 1.0)))

    # Ablation rows B and A of spec §2.4, expressed as routing degeneracies rather
    # than as separate code paths, so they share this exact pipeline.
    fa = float(getattr(cfg, "force_alpha", -1.0))
    force_alpha = fa if fa >= 0 else None     # dataclasses cannot hold None here
    active_frac = float(getattr(cfg, "active_frac", 0.6))
    if policy == "prune_only":          # M_t = empty -> pure pruning
        force_alpha = 1.0
    elif policy == "merge_only":        # D_t = empty -> pure merging
        active_frac = 1.0

    rt = route_tokens(
        sc["S"], sc["R_raw_mean"], b_t,
        alpha_min=float(getattr(cfg, "alpha_min", 0.4)),
        alpha_max=float(getattr(cfg, "alpha_max", 0.8)),
        beta=None if active_frac is not None else float(getattr(cfg, "beta", 4.0)),
        active_frac=active_frac,
        force_alpha=force_alpha,
        alpha_flip=bool(getattr(cfg, "alpha_flip", False)),
    )

    merged = []
    for t in range(L):
        merged.append(seeded_merge(video_features[t], rt["pool_idx"][t],
                                   sc["S"][t], int(rt["B_M"][t])))

    tokens, g = assemble(video_features, rt["retain_idx"], merged=merged,
                         expected_total=B)
    flashvid_config.visual_token_length = int(tokens.shape[0])
    return tokens, g


def no_llm_pruning(hidden_states, causal_mask, attentions, cache_position,
                   position_ids, position_embeddings, flashvid_config,
                   visual_pos_masks=None):
    """Inner-LLM pruning stage for method ``bv``: keep everything.

    FlashVID carries a SECOND, independent budget -- the vision side keeps
    `retention_ratio * expansion` of the tokens and layer `pruning_layer` then
    cuts to `llm_retention_ratio` of what survived, which is why its headline
    "R" is a per-layer average and its true visual-token count sits ~30% above
    the naive r*N (experiments/flashvid_token_accounting).

    This method has one budget by construction: B is the number of tokens the
    LLM is given, and §2.2's budget equality is exact. Pruning again inside the
    LLM would make the reported B a lie. So this is a deliberate no-op, not a
    stub -- keep_indices is every position, and nothing else is touched.
    """
    keep = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    if cache_position is None:
        cache_position = keep
    if position_ids is None:
        position_ids = keep.unsqueeze(0)
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep
