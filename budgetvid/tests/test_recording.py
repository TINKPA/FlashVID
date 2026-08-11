"""Recording-path tests: dumps written by the pipeline must be complete and
consistent with the routing they describe. No GPU, no model, no decord.

    uv run --no-project --python 3.11 --with torch --with numpy python budgetvid/tests/test_recording.py
"""

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.adapters.pipeline import budgetvid_pipeline  # noqa: E402
from budgetvid.core.merging import seeded_merge              # noqa: E402

L, N_F, D = 4, 16, 8   # 4x4 grid
R = 0.25               # B = 16, b_t = 4
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def make_cfg(tmp, policy="threeway", **kw):
    base = dict(policy=policy, retention_ratio=R, seed=42,
                eta=0.5, lam=1.0, alpha_min=0.4, alpha_max=0.8,
                active_frac=0.6, alpha_flip=False, force_alpha=-1.0,
                debias_pos=False, dump_dir=str(tmp), visual_token_length=0)
    base.update(kw)
    return SimpleNamespace(**base)


torch.manual_seed(0)
X = torch.randn(L, N_F, D)
I = torch.rand(L, N_F)
B = int(round(R * L * N_F))

print("threeway dump:")
with tempfile.TemporaryDirectory() as tmp:
    c = make_cfg(tmp)
    c.dump_tag = "vid_a"
    tokens, g = budgetvid_pipeline(X, I, c)
    p = pathlib.Path(tmp) / "vid_a.npz"
    check("npz written", p.exists())
    z = np.load(p)
    keys = {"I_raw", "I_used", "R_sp", "R_tp", "R_raw", "I_hat", "R_hat", "S",
            "alpha", "labels", "seed_of", "kept_g", "b_t", "B_R", "B_M",
            "N_active", "meta"}
    check("all keys present", keys <= set(z.files), str(sorted(set(z.files) - keys)))
    labels, b_t = z["labels"], z["b_t"]
    B_R, B_M, N_act = z["B_R"], z["B_M"], z["N_active"]
    check("labels shape", labels.shape == (L, N_F))
    check("retain count == B_R",
          all(int((labels[t] == 2).sum()) == int(B_R[t]) for t in range(L)))
    check("merge-pool count == N_active - B_R (or 0 when B_M=0)",
          all(int((labels[t] == 1).sum()) ==
              (int(N_act[t] - B_R[t]) if B_M[t] > 0 else 0) for t in range(L)))
    check("kept_g length == B", z["kept_g"].shape[0] == B, str(z["kept_g"].shape))
    check("kept_g matches pipeline output",
          np.array_equal(z["kept_g"], g.cpu().numpy().astype(np.int32)))

    seed_of = z["seed_of"]
    ok = True
    for t in range(L):
        pool = np.where(labels[t] == 1)[0]
        ok &= np.array_equal(np.sort(np.where(seed_of[t] >= 0)[0]), pool)
        ok &= all(labels[t][seed_of[t][tok]] == 1 for tok in pool)
    check("seed_of covers exactly the pool and points into it", bool(ok))

    seeds = {t * N_F + int(seed_of[t][tok])
             for t in range(L) for tok in np.where(labels[t] == 1)[0]}
    retained = {t * N_F + int(i)
                for t in range(L) for i in np.where(labels[t] == 2)[0]}
    check("kept_g == retained U seed positions",
          sorted(seeds | retained) == sorted(z["kept_g"].tolist()))

    meta = json.loads(str(z["meta"]))
    check("meta records config",
          meta["lam"] == 1.0 and meta["eta"] == 0.5
          and meta["policy"] == "threeway" and meta["grid"] == [4, 4])

    sj = pathlib.Path(tmp) / "summary.jsonl"
    check("summary.jsonl written",
          sj.exists() and len(sj.read_text().strip().splitlines()) == 1)
    line = json.loads(sj.read_text().strip())
    check("summary has distribution stats",
          {"I_entropy_norm", "I_top10_mass", "I_max_over_mean",
           "corr_Rsp_Rtp"} <= set(line))

    tokens2, g2 = budgetvid_pipeline(X, I, c)
    check("dedupe: still one summary line",
          len(sj.read_text().strip().splitlines()) == 1)
    check("determinism: identical kept_g on re-run", torch.equal(g, g2))

print("random_drop dump:")
with tempfile.TemporaryDirectory() as tmp:
    c = make_cfg(tmp, policy="random_drop")
    c.dump_tag = "vid_b"
    tokens, g = budgetvid_pipeline(X, I, c)
    z = np.load(pathlib.Path(tmp) / "vid_b.npz")
    check("kept labels count == B", int((z["labels"] == 2).sum()) == B)
    check("no merge labels", int((z["labels"] == 1).sum()) == 0)
    check("minimal keys", {"I_raw", "labels", "kept_g", "b_t", "meta"} <= set(z.files))

print("non-square grid (2x8) via config H/W:")
with tempfile.TemporaryDirectory() as tmp:
    c = make_cfg(tmp, H=2, W=8)
    c.dump_tag = "vid_c"
    tokens, g = budgetvid_pipeline(X, I, c)
    z = np.load(pathlib.Path(tmp) / "vid_c.npz")
    check("grid recorded as [2, 8]", json.loads(str(z["meta"]))["grid"] == [2, 8])
    check("budget still exact", z["kept_g"].shape[0] == B)

print("seeded_merge return_assign:")
S_frame = torch.rand(N_F)
m, s = seeded_merge(X[0], torch.arange(6), S_frame, 3)
m2, s2, pt, ps = seeded_merge(X[0], torch.arange(6), S_frame, 3, return_assign=True)
check("merge unchanged by return_assign", torch.equal(m, m2) and torch.equal(s, s2))
check("assignment covers the whole pool", pt.numel() == 6 and ps.numel() == 6)
check("seeds map to themselves", torch.equal(ps[:3], pt[:3]))
em, es, ept, eps = seeded_merge(X[0], torch.arange(0), S_frame, 0, return_assign=True)
check("empty-pool edge returns empties", em.shape[0] == 0 and ept.numel() == 0)
sm, ss, spt, sps = seeded_merge(X[0], torch.arange(3), S_frame, 3, return_assign=True)
check("rest-empty edge: pool == seeds", torch.equal(spt, sps) and spt.numel() == 3)

print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
sys.exit(1 if FAIL else 0)
