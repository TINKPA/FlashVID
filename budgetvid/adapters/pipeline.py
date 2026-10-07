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

from ..core.assembly import assemble
from ..core.budget import split_budget
from ..core.quantize import compress_video, curve_cost
from ..mass_bias import clear_mass
from ..recording import dump_record, frame_stats, make_tag

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
    out = compress_video(
        video_features, B, W_k=W_k, W_v=W_v, g=g,
        gamma_v=float(getattr(cfg, "gamma_v", 1.0)),
        alloc=str(getattr(cfg, "mq_alloc", "waterfill")),
        centroid=str(getattr(cfg, "centroid", "rms")),
        b_max=int(getattr(cfg, "b_max", 0)),
        refine=int(getattr(cfg, "refine", 0)),
    )
    spent = int(out["b"].sum())
    empty = [torch.empty(0, dtype=torch.long, device=video_features.device)] * L
    merged = list(zip(out["feats"], out["seed_idx"]))
    tokens, gidx, mass = assemble(video_features, empty, merged=merged,
                                  expected_total=spent, masses=out["mass"])
    cfg.visual_token_length = int(tokens.shape[0])
    # The mass channel. Off is the mandatory ablation (a conventional
    # mass-destroying merge), and it must travel this same code path so the two
    # rows differ in one thing only.
    cfg.token_mass = mass if bool(getattr(cfg, "mass", True)) else None

    dump_dir = str(getattr(cfg, "dump_dir", "") or "")
    if dump_dir:
        # The counterfactual allocation costs nothing to evaluate: D_t(b) is the
        # realized cost at every size, so the allocation we did NOT take is a
        # lookup on curves we already have. This is what answers P2 of the
        # offline-replay note (does water-filling beat the even split on
        # quantization cost, and by how much) as a by-product of the run itself.
        alloc = str(getattr(cfg, "mq_alloc", "waterfill"))
        if alloc == "video":
            # One video-level curve, not per-frame ones: there is no per-frame
            # allocation to price, so both lookups are just the realized cost.
            cost_even = cost_wf = out["cost"]
        else:
            b_alt = split_budget(B, L, N_f).to(out["b"].device) \
                if alloc == "waterfill" else out["b"]
            cost_even = curve_cost(out["D"], b_alt)
            cost_wf = curve_cost(out["D"], out["b"])
        b_full = torch.zeros(L, N_f, dtype=torch.int32)
        for t, mm in enumerate(out["mass"]):
            b_full[t, out["seed_idx"][t].cpu().long()] = mm.cpu().to(torch.int32)
        dump_record(dump_dir, make_tag(cfg),
                    {"I_raw": cls_attention, "radius": out["radius"],
                     "D": out["D"], "r_curve": out["r"]},
                    {"mass_map": b_full, "kept_g": gidx, "b_t": out["b"],
                     "b_even": b_t, "group_of": out["group_of"]},
                    {"method": "bv", "policy": "mq", "L": L, "N_f": N_f,
                     "retention_ratio": float(cfg.retention_ratio), "B": B,
                     "B_spent": spent,
                     "lift": str(getattr(cfg, "lift", "kv")),
                     "lift_norm": bool(getattr(cfg, "lift_norm", True)),
                     "gamma_v": float(getattr(cfg, "gamma_v", 1.0)),
                     "mq_alloc": str(getattr(cfg, "mq_alloc", "waterfill")),
                     "centroid": str(getattr(cfg, "centroid", "rms")),
                     "refine": int(getattr(cfg, "refine", 0)),
                     "mass": bool(getattr(cfg, "mass", True)),
                     "cost": out["cost"], "planned": out["planned"],
                     "cost_taken": cost_wf, "cost_even": cost_even,
                     "spec": "2026-08-28_method_budgetvid2_v1"},
                    frame_stats(cls_attention), cfg=cfg)
    return tokens, gidx


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
