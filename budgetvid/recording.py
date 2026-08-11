"""Per-video recording of signals and routing decisions (Round 1, spec v1.3).

The round's goal is insight, not scores: every number in a results table must
be traceable to token level. This module dumps, per video:

* signals  -- ``I_raw`` (and ``I_used`` when de-biased), ``R_sp``, ``R_tp``,
  ``R_raw``, ``I_hat``, ``R_hat``, ``S``, all ``[L, N_f]`` fp16;
* routing  -- per-token ``labels`` (0 drop / 1 merge / 2 retain), ``seed_of``
  (the within-frame seed each pool token was absorbed into, -1 elsewhere),
  ``kept_g`` (global indices that actually entered the LLM), and the
  per-frame vectors ``alpha``, ``b_t``, ``B_R``, ``B_M``, ``N_active``;
* ``summary.jsonl`` -- one line per video of cheap per-frame distribution
  stats (normalised entropy, top-10% mass, max/mean of I; corr(R_sp, R_tp)),
  so distribution plots don't require unpacking every npz.

One ``<tag>.npz`` per video per run directory; ``tag`` is set by the eval
wrapper from the video filename (``dump_tag`` on the config). The method is
query-agnostic and deterministic (verified in Step 1), so a video that
appears under several questions is dumped once -- existing files are never
overwritten, and the write is temp-file + atomic rename so concurrent ranks
that race on the same video cannot interleave.

Overlays MUST be rendered from these dumps, never recomputed with
script-side knobs: the 2026-08-04 overlay mismatch (figures drawn at beta=8
against runs that used active_frac=0.6) is the failure mode this rule closes.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch


def _f16(x: torch.Tensor) -> np.ndarray:
    return x.detach().to(torch.float32).cpu().numpy().astype(np.float16)


def _i32(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy().astype(np.int32)


def make_tag(cfg) -> str:
    """The per-video dump tag: wrapper-set filename stem, or a unique fallback.

    The fallback embeds pid + monotonic ns so distributed ranks can never
    collide on a name; joining such records back to questions then goes
    through file order, which is why the wrapper should always set dump_tag.
    """
    tag = getattr(cfg, "dump_tag", None)
    if tag:
        return str(tag)
    return f"sample_{os.getpid()}_{time.monotonic_ns()}"


def frame_stats(I: torch.Tensor,
                R_sp: torch.Tensor | None = None,
                R_tp: torch.Tensor | None = None) -> dict:
    """Cheap per-frame distribution stats of the importance signal.

    Same quantities as the Step 1 non-uniformity gate (normalised entropy,
    top-10% mass, max/mean), plus per-frame corr(R_sp, R_tp) when the
    redundancy maps are given -- the H2 statistic, kept per video so the
    distribution can be re-plotted without opening npz files.
    """
    I = I.detach().float().cpu()
    p = I / I.sum(-1, keepdim=True).clamp_min(1e-12)
    ent = -(p * p.clamp_min(1e-12).log()).sum(-1) / math.log(max(I.shape[-1], 2))
    k = max(1, I.shape[-1] // 10)
    top = p.topk(k, dim=-1).values.sum(-1)
    maxmean = I.max(-1).values / I.mean(-1).clamp_min(1e-12)
    out = {
        "I_entropy_norm": [round(v, 4) for v in ent.tolist()],
        "I_top10_mass": [round(v, 4) for v in top.tolist()],
        "I_max_over_mean": [round(v, 2) for v in maxmean.tolist()],
    }
    if R_sp is not None and R_tp is not None:
        rs = R_sp.detach().float().cpu()
        rt = R_tp.detach().float().cpu()
        rs_c = rs - rs.mean(-1, keepdim=True)
        rt_c = rt - rt.mean(-1, keepdim=True)
        corr = (rs_c * rt_c).sum(-1) / (rs_c.norm(dim=-1) * rt_c.norm(dim=-1)).clamp_min(1e-12)
        out["corr_Rsp_Rtp"] = [round(v, 4) for v in corr.tolist()]
    return out


def dump_record(dump_dir: str, tag: str,
                float_arrays: dict[str, torch.Tensor],
                int_arrays: dict[str, torch.Tensor],
                meta: dict,
                stats: dict | None) -> str | None:
    """Write ``<dump_dir>/<tag>.npz`` and a summary.jsonl line.

    Returns the path, or None when the record already exists (first dump of a
    video wins; content is deterministic so later calls carry nothing new).
    """
    os.makedirs(dump_dir, exist_ok=True)
    path = os.path.join(dump_dir, f"{tag}.npz")
    if os.path.exists(path):
        return None
    payload: dict[str, np.ndarray] = {k: _f16(v) for k, v in float_arrays.items()}
    payload.update({k: _i32(v) for k, v in int_arrays.items()})
    payload["meta"] = np.array(json.dumps(meta))
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **payload)
    os.replace(tmp, path)
    if stats is not None:
        line = {"tag": tag, **meta, **stats}
        with open(os.path.join(dump_dir, "summary.jsonl"), "a") as f:
            f.write(json.dumps(line) + "\n")
    return path


_FLASHVID_WRAPPED = False


def wrap_flashvid_keep(dump_dir: str) -> None:
    """Shim the registered "flashvid" compression entry to record its choices.

    FlashVID's own kept indices are dumped so case analysis can ask "the token
    we dropped and they kept -- was it the one that mattered?". The shim lives
    in the dispatch registry; no upstream file changes (fork rule 3: keep the
    upstream diff minimal).
    """
    global _FLASHVID_WRAPPED
    if _FLASHVID_WRAPPED:
        return
    from flashvid import dispatch

    orig = dispatch._COMPRESSION_REGISTRY["flashvid"]

    def recording_flashvid(video_features, cls_attention, flashvid_config):
        tokens, g = orig(video_features=video_features, cls_attention=cls_attention,
                         flashvid_config=flashvid_config)
        dump_record(
            dump_dir, make_tag(flashvid_config),
            {"I_raw": cls_attention},
            {"kept_g": g},
            {"method": "flashvid",
             "L": int(video_features.shape[0]), "N_f": int(video_features.shape[1]),
             "retention_ratio": float(getattr(flashvid_config, "retention_ratio", -1.0))},
            frame_stats(cls_attention),
        )
        return tokens, g

    dispatch.register_compression("flashvid", recording_flashvid)
    _FLASHVID_WRAPPED = True
