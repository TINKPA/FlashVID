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

import hashlib
import pathlib

import torch

from ..core.assembly import assemble
from ..core.budget import split_budget
from ..core.merging import seeded_merge
from ..core.policies import attn_top_keep, random_drop_keep, uniform_keep
from ..core.quantize import compress_video, curve_cost
from ..core.routing import route_tokens
from ..core.scoring import score_tokens
from ..mass_bias import clear_mass
from ..recording import dump_record, frame_stats, make_tag

POLICIES = ("none", "random_drop", "uniform", "threeway", "prune_only", "merge_only",
            "attn_top", "attn_top_nosink", "mq")

_ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"


def _rank01(v: torch.Tensor) -> torch.Tensor:
    """Within-frame rank percentile, matching core.scoring.rank_normalize."""
    n = v.shape[-1]
    o = v.argsort(dim=-1, stable=True)
    r = torch.empty_like(o)
    r.scatter_(-1, o, torch.arange(n, device=v.device).expand_as(o))
    return r.float() / max(n - 1, 1)


def _derived_seed(base_seed: int, tag: str) -> int:
    """Per-video seed for random policies (Round-2 preprocessing step 3).

    Round 1 seeded every video with the same global 42, so random_drop used ONE
    mask for the whole benchmark -- coverage variance across videos was zero and
    per-video analysis on the random arm was structurally impossible. Deriving
    from (base_seed, video tag) keeps runs reproducible while making the masks
    independent across videos.
    """
    return int(hashlib.sha256(f"{base_seed}:{tag}".encode()).hexdigest()[:8], 16)


def _grid_from_config(cfg, N_f: int):
    """(H, W) if the modeling code declared the native grid and it matches N_f."""
    h = int(getattr(cfg, "H", 0) or 0)
    w = int(getattr(cfg, "W", 0) or 0)
    if h > 0 and w > 0 and h * w == N_f:
        return h, w
    return None


def _sink_cells(cfg, N_f: int) -> tuple[torch.Tensor, str]:
    """Load the offline sink-cell set for the video's native grid.

    Built by experiments/budgetvid/scripts/build_qwen3_pos_baseline.py from the
    Round-1 dumps (mean within-frame rank >= 0.85, corner + border population;
    provenance in assets/pos_baseline_qwen3_meta.json). attn_top_nosink cannot
    run without it -- that is by design, not a fallback situation.
    """
    grid = _grid_from_config(cfg, N_f)
    if grid is None:
        raise RuntimeError(
            "attn_top_nosink needs the native grid (config.H/W); it is only "
            "wired for backbones that declare it (Qwen3-VL)")
    p = _ASSETS / f"sink_cells_qwen3_{grid[0]}x{grid[1]}.npy"
    if not p.exists():
        raise FileNotFoundError(
            f"positional baseline not built for grid {grid[0]}x{grid[1]}: {p} "
            "(run build_qwen3_pos_baseline.py first)")
    import numpy as np
    return torch.from_numpy(np.load(p)).long(), p.name


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

    seed = int(getattr(flashvid_config, "seed", 42))
    if policy in ("uniform", "random_drop", "attn_top", "attn_top_nosink"):
        extra_meta = {}
        if policy == "uniform":
            keep = uniform_keep(L, N_f, b_t, device=device)
        elif policy == "random_drop":
            tag = str(getattr(flashvid_config, "dump_tag", "") or "")
            if tag:
                extra_meta = {"seed_base": seed, "seed_tag": tag}
                seed = _derived_seed(seed, tag)
            keep = random_drop_keep(L, N_f, b_t, seed=seed, device=device)
        else:
            sink = None
            if policy == "attn_top_nosink":
                sink, asset = _sink_cells(flashvid_config, N_f)
                extra_meta = {"n_sink_cells": int(sink.numel()), "sink_asset": asset}
            keep = [k.to(device) for k in attn_top_keep(cls_attention, b_t,
                                                        sink_cells=sink)]
        for t, k in enumerate(keep):
            assert k.numel() == int(b_t[t]), (t, k.numel(), int(b_t[t]))
        tokens, g = assemble(video_features, keep, merged=None, expected_total=B)
        flashvid_config.visual_token_length = int(tokens.shape[0])
        dump_dir = str(getattr(flashvid_config, "dump_dir", "") or "")
        if dump_dir:
            labels = torch.zeros(L, N_f, dtype=torch.int8)
            for t, k in enumerate(keep):
                labels[t, k.cpu().long()] = 2
            dump_record(dump_dir, make_tag(flashvid_config),
                        {"I_raw": cls_attention},
                        {"labels": labels, "kept_g": g, "b_t": b_t},
                        {"method": "bv", "policy": policy, "L": L, "N_f": N_f,
                         "retention_ratio": r, "B": B, "seed": seed, **extra_meta},
                        frame_stats(cls_attention))
        return tokens, g

    if policy == "mq":
        return _measure_quantization(video_features, cls_attention, flashvid_config,
                                     b_t, B, L, N_f)

    # ---- v0: score -> route -> merge -> assemble ----
    cfg = flashvid_config
    # Grid shape: Qwen3-VL's native-resolution grid is not square; its modeling
    # code stores (H, W) on the config before compress() (spec v1.3 §2.1). The
    # LLaVA path has no such fields and stays on the square-root route.
    cfg_h = int(getattr(cfg, "H", 0) or 0)
    cfg_w = int(getattr(cfg, "W", 0) or 0)
    if cfg_h > 0 and cfg_w > 0 and cfg_h * cfg_w == N_f:
        grid = (cfg_h, cfg_w)
    else:
        s = int(round(N_f ** 0.5))
        if s * s != N_f:
            raise ValueError(
                f"N_f={N_f} is not square and config H*W ({cfg_h}x{cfg_w}) does not "
                "match it; the 4-neighbourhood needs the true grid shape")
        grid = (s, s)

    I_used = cls_attention.float()
    if bool(getattr(cfg, "debias_pos", False)):
        # The importance signal is ~27% positional (Step 2): eight of the 196
        # cells sit in the frame's top-b_t 98.5% of the time. At b_t=2 that means
        # both retained tokens are the same two grid cells in every frame, so a
        # 256-frame video contributes 256 copies of one position instead of
        # coverage. Subtracting the position mean removes exactly that component
        # and leaves the content-dependent part.
        import numpy as _np
        # Native-grid backbones (Qwen3-VL) get a per-grid baseline built offline
        # from the Round-1 dumps; the LLaVA square grid keeps its original file.
        _pq = _ASSETS / f"pos_baseline_qwen3_{grid[0]}x{grid[1]}.npy"
        if _grid_from_config(cfg, N_f) is not None and _pq.exists():
            _b = _np.load(_pq).reshape(-1)
        else:
            _b = _np.load(_ASSETS / f"pos_baseline_pooled{N_f}.npy")
        base = torch.from_numpy(_b).to(I_used.device, I_used.dtype)
        I_used = _rank01(I_used) - base.unsqueeze(0)

    sc = score_tokens(video_features.float(), I_used, grid,
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

    merged, absorb = [], []
    for t in range(L):
        m, s, pool_tok, pool_seed = seeded_merge(video_features[t], rt["pool_idx"][t],
                                                 sc["S"][t], int(rt["B_M"][t]),
                                                 return_assign=True)
        merged.append((m, s))
        absorb.append((pool_tok, pool_seed))

    tokens, g = assemble(video_features, rt["retain_idx"], merged=merged,
                         expected_total=B)
    flashvid_config.visual_token_length = int(tokens.shape[0])

    dump_dir = str(getattr(cfg, "dump_dir", "") or "")
    if dump_dir:
        _dump_threeway(dump_dir, cfg, policy, r, B, grid, cls_attention, I_used,
                       sc, rt, absorb, b_t, g, L, N_f)
    return tokens, g


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
        b_alt = split_budget(B, L, N_f).to(out["b"].device) \
            if str(getattr(cfg, "mq_alloc", "waterfill")) == "waterfill" else out["b"]
        cost_even = curve_cost(out["D"], b_alt)
        cost_wf = curve_cost(out["D"], out["b"])
        b_full = torch.zeros(L, N_f, dtype=torch.int32)
        for t, mm in enumerate(out["mass"]):
            b_full[t, out["seed_idx"][t].cpu().long()] = mm.cpu().to(torch.int32)
        dump_record(dump_dir, make_tag(cfg),
                    {"I_raw": cls_attention, "radius": out["radius"],
                     "D": out["D"], "r_curve": out["r"]},
                    {"mass_map": b_full, "kept_g": gidx, "b_t": out["b"],
                     "b_even": b_t},
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
                    frame_stats(cls_attention))
    return tokens, gidx


def _dump_threeway(dump_dir, cfg, policy, r, B, grid, cls_attention, I_used,
                   sc, rt, absorb, b_t, g, L, N_f):
    """Round-1 recording: routing labels, absorption map, and every scoring
    intermediate, per video. See budgetvid/recording.py for the file layout."""
    labels = torch.zeros(L, N_f, dtype=torch.int8)
    seed_of = torch.full((L, N_f), -1, dtype=torch.int32)
    for t in range(L):
        labels[t, rt["retain_idx"][t].cpu().long()] = 2
        p = rt["pool_idx"][t]
        if p.numel():
            labels[t, p.cpu().long()] = 1
        pool_tok, pool_seed = absorb[t]
        if pool_tok.numel():
            seed_of[t, pool_tok.cpu().long()] = pool_seed.cpu().to(torch.int32)
    meta = {
        "method": "bv", "policy": policy, "L": L, "N_f": N_f, "grid": list(grid),
        "retention_ratio": r, "B": B,
        "lam": float(getattr(cfg, "lam", 1.0)),
        "eta": float(getattr(cfg, "eta", 0.5)),
        "alpha_min": float(getattr(cfg, "alpha_min", 0.4)),
        "alpha_max": float(getattr(cfg, "alpha_max", 0.8)),
        "active_frac": float(getattr(cfg, "active_frac", 0.6)),
        "alpha_flip": bool(getattr(cfg, "alpha_flip", False)),
        "debias_pos": bool(getattr(cfg, "debias_pos", False)),
        "force_alpha": float(getattr(cfg, "force_alpha", -1.0)),
        "seed": int(getattr(cfg, "seed", 42)),
    }
    floats = {"I_raw": cls_attention, "I_used": I_used,
              "R_sp": sc["R_sp"], "R_tp": sc["R_tp"], "R_raw": sc["R_raw"],
              "I_hat": sc["I_hat"], "R_hat": sc["R_hat"], "S": sc["S"],
              "alpha": rt["alpha"]}
    ints = {"labels": labels, "seed_of": seed_of, "kept_g": g,
            "b_t": b_t, "B_R": rt["B_R"], "B_M": rt["B_M"],
            "N_active": rt["N_active"]}
    dump_record(dump_dir, make_tag(cfg), floats, ints, meta,
                frame_stats(cls_attention, sc["R_sp"], sc["R_tp"]))


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
