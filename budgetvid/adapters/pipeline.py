"""Registry entry point for method ``bv``: one policy switch, one assembly path.

Which policy runs is ``config.policy``: ``"none"`` passes every token through
(the vanilla row), ``"mq"`` is MassVID (measure quantization, spec
notes/2026-08-28_method_budgetvid2_v1.html). Everything after the policy is
shared, which is what makes an ablation row comparable to a baseline row rather
than a different program that happens to produce a similar number.

Budget is an ABSOLUTE token count, per the spec: B = round(r * N) with N = L*N_f.
Note this differs from FlashVID's own `retention_ratio`, which is a per-LLM-layer
average and lands ~30% higher in actual visual tokens at the LLM input
(experiments/flashvid_token_accounting). Do not read the two as the same knob.

The Round-1 policies (score -> route -> merge; uniform / random_drop / attn_top
baselines) were removed on 2026-10-07; every run that used them records its
commit, and 89c59fe is the last one that carries them.
"""

from __future__ import annotations

import torch

from ..core.assembly import assemble, reconstruct_full, reconstruct_rank
from ..core.budget import split_budget
from ..core.quantize import compress_video
from ..llm_prune import no_llm_pruning, reset_inner_state  # noqa: F401 (re-export)
from ..mass_bias import clear_mass

POLICIES = ("none", "mq")


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
    # Every sample starts with no mass: a stale vector from the previous video
    # would bias the wrong keys and never raise.
    clear_mass(flashvid_config)
    reset_inner_state(flashvid_config)

    if policy == "none":
        g = torch.arange(L * N_f, dtype=torch.long, device=device)
        flashvid_config.visual_token_length = L * N_f
        return video_features.reshape(L * N_f, -1), g

    if policy not in POLICIES:
        raise KeyError(f"unknown policy '{policy}'; known: {sorted(POLICIES)}")

    r = float(flashvid_config.retention_ratio)
    B = int(round(r * L * N_f))
    b_t = split_budget(B, L, N_f).to(device)
    return _measure_quantization(video_features, cls_attention, flashvid_config,
                                 b_t, B, L, N_f)


def _lift_params(cfg, device, dtype):
    """The frozen projections Phi needs, as captured by ``budgetvid()``.

    Returns ``(W_k, W_v, g)``, any of which may be None: ``lift="none"`` measures
    in the projector space (the metric ablation), ``lift="key"`` drops the value
    half, ``lift_norm=False`` drops the RMSNorm gain.
    """
    mode = str(getattr(cfg, "lift", "kv"))
    if mode == "none":
        return None, None, None
    lp = getattr(cfg, "lift_params", None)
    if not lp:
        raise RuntimeError(
            "policy='mq' needs the decoder's layer-0 k_proj/v_proj/input_layernorm, "
            "which budgetvid() captures at patch time. None were found -- either "
            "the model was patched by flashvid() directly, or its first decoder "
            "layer does not expose k_proj (see budgetvid/__init__.py:_capture_lift).")
    W_k = lp["W_k"].to(device)
    W_v = lp["W_v"].to(device) if mode != "key" else None
    g = lp["g"].to(device) if bool(getattr(cfg, "lift_norm", True)) else None
    return W_k, W_v, g


def _measure_quantization(video_features, cls_attention, cfg, b_t, B, L, N_f):
    """BudgetVID 2.0: CBA allocation + mass-preserving quantization.

    The per-frame split ``b_t`` computed by the caller is NOT used when
    ``mq_alloc="waterfill"`` -- CBA replaces it -- but it is what
    ``mq_alloc="even"`` falls back to, which is exactly the v0 allocation and so
    isolates what the allocation alone is worth.
    """
    W_k, W_v, g = _lift_params(cfg, video_features.device, video_features.dtype)
    want_eff = bool(getattr(cfg, "mass_eff", False))
    out = compress_video(
        video_features, B, W_k=W_k, W_v=W_v, g=g,
        gamma_v=float(getattr(cfg, "gamma_v", 1.0)),
        alloc=str(getattr(cfg, "mq_alloc", "waterfill")),
        centroid=str(getattr(cfg, "centroid", "rms")),
        b_max=int(getattr(cfg, "b_max", 0)),
        refine=int(getattr(cfg, "refine", 0)),
        protect=float(getattr(cfg, "protect", 0.0)),
        mass_eff=want_eff,
    )
    spent = int(out["b"].sum())
    empty = [torch.empty(0, dtype=torch.long, device=video_features.device)] * L
    merged = list(zip(out["feats"], out["seed_idx"]))
    tokens, gidx, mass = assemble(video_features, empty, merged=merged,
                                  expected_total=spent, masses=out["mass"])

    recon = str(getattr(cfg, "reconstruct", "none"))
    if recon != "none":
        # Full-length re-expansion (oracle rows, see configuration_budgetvid.py).
        # Emitting arange(L*N_f) as the global indices makes the backbone's
        # scatter/gather/position filtering the identity: every copy sits at its
        # member's original position id. The mass channel is forced off because
        # multiplicity is physical again -- biasing on top would double-count.
        if str(getattr(cfg, "llm_prune", "none")) != "none":
            raise ValueError("reconstruct requires llm_prune='none': the inner-LLM "
                             "stage's visual bookkeeping assumes the compressed span")
        if recon == "group":
            rr = int(getattr(cfg, "recon_rank", 0))
            full = (reconstruct_rank(tokens, gidx, out["group_of"], video_features, rr)
                    if rr > 0 else reconstruct_full(tokens, gidx, out["group_of"]))
        elif recon == "identity":
            full = video_features.reshape(L * N_f, -1)
        else:
            raise KeyError(f"unknown reconstruct '{recon}'; known: none, group, identity")
        g_full = torch.arange(L * N_f, dtype=torch.long, device=video_features.device)
        cfg.visual_token_length = L * N_f
        cfg.token_mass = None
        cfg.token_mass_eff = None
        return full.to(video_features.dtype), g_full

    cfg.visual_token_length = int(tokens.shape[0])
    # The mass channel. Off is the mandatory ablation (a conventional
    # mass-destroying merge), and it must travel this same code path so the two
    # rows differ in one thing only.
    cfg.token_mass = mass if bool(getattr(cfg, "mass", True)) else None
    cfg.token_mass_eff = None
    if want_eff and cfg.token_mass is not None:
        # Same order as the tokens: replicate assemble()'s sort (per-frame concat
        # of the seeds' global indices, then argsort). Asserted against gidx so
        # a divergence fails loudly rather than biasing the wrong keys.
        g_cat = torch.cat([t * N_f + out["seed_idx"][t].to(torch.long) for t in range(L)])
        order = g_cat.argsort()
        if not torch.equal(g_cat[order], gidx):
            raise AssertionError("mass_eff ordering diverged from assemble()'s")
        cfg.token_mass_eff = torch.cat(out["mass_eff"], 0)[order]

    return tokens, gidx

