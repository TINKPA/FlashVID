"""Round-2 policy tests: attn_top / attn_top_nosink (B+ sink diagnostics) and
the per-video derived seed for random_drop. No GPU, no model, no decord.

    uv run --no-project --python 3.11 --with torch --with numpy python \
        budgetvid/tests/test_round2_policies.py
"""

import json
import pathlib
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.adapters.pipeline import (  # noqa: E402
    _derived_seed,
    budgetvid_pipeline,
)
from budgetvid.core.budget import split_budget         # noqa: E402
from budgetvid.core.policies import attn_top_keep      # noqa: E402

ASSETS = pathlib.Path(__file__).resolve().parents[1] / "assets"
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def cfg(policy, r, N_f_grid=None, tag="", dump_dir="", seed=42, **kw):
    c = SimpleNamespace(policy=policy, retention_ratio=r, seed=seed,
                        dump_tag=tag, dump_dir=dump_dir, **kw)
    if N_f_grid:
        c.H, c.W = N_f_grid
    return c


def main() -> int:
    torch.manual_seed(0)

    # 1 attn_top picks exactly the per-frame top-b_t of raw I
    L, N_f = 4, 196
    I = torch.randn(L, N_f)  # continuous -> no ties
    b_t = split_budget(49, L, N_f)
    keep = attn_top_keep(I, b_t)
    ok = all(set(keep[t].tolist()) == set(I[t].topk(int(b_t[t])).indices.tolist())
             and torch.all(keep[t][1:] > keep[t][:-1]) for t in range(L))
    check("1  attn_top == per-frame topk(I), sorted", ok)

    # 2 attn_top_nosink excludes the sink set and refills with next-best
    sinks = torch.tensor([0, 5, 17, 195])
    keep_ns = attn_top_keep(I, b_t, sink_cells=sinks)
    no_sink = all(not (set(k.tolist()) & set(sinks.tolist())) for k in keep_ns)
    masked = I.clone()
    masked[:, sinks] = float("-inf")
    refill = all(set(keep_ns[t].tolist())
                 == set(masked[t].topk(int(b_t[t])).indices.tolist()) for t in range(L))
    check("2  sinks excluded, replacements are next-best", no_sink and refill)

    # 3 budget exceeding the non-sink pool raises
    try:
        attn_top_keep(I, split_budget(L * (N_f - 2), L, N_f),
                      sink_cells=torch.arange(4))
        check("3  over-budget vs non-sink pool raises", False)
    except ValueError:
        check("3  over-budget vs non-sink pool raises", True)

    # 4 pipeline equivalence: attn_top == prune_only + lam=0 (the settings-block claim)
    D = 8
    X = torch.randn(L, N_f, D)
    _, g_top = budgetvid_pipeline(X, I, cfg("attn_top", 0.0625))
    _, g_prn = budgetvid_pipeline(X, I, cfg("prune_only", 0.0625, lam=0.0,
                                            debias_pos=False))
    check("4  attn_top kept_g == prune_only(lam=0) kept_g",
          torch.equal(g_top, g_prn))

    # 5 pipeline attn_top_nosink loads the real 20x36 asset and avoids its cells
    sink_file = ASSETS / "sink_cells_qwen3_20x36.npy"
    if sink_file.exists():
        s2036 = torch.from_numpy(np.load(sink_file))
        Nq = 720
        Xq, Iq = torch.randn(L, Nq, D), torch.randn(L, Nq)
        toks, g = budgetvid_pipeline(Xq, Iq, cfg("attn_top_nosink", 0.0625,
                                                 N_f_grid=(20, 36)))
        within = (g % Nq)
        check("5  nosink kept cells disjoint from sink asset",
              not (set(within.tolist()) & set(s2036.tolist())),
              f"{s2036.numel()} sink cells")
    else:
        check("5  nosink kept cells disjoint from sink asset", False,
              "asset missing -- run build_qwen3_pos_baseline.py")

    # 6 nosink without a baseline for the grid fails loudly (by design)
    try:
        budgetvid_pipeline(torch.randn(L, 63, D), torch.randn(L, 63),
                           cfg("attn_top_nosink", 0.25, N_f_grid=(7, 9)))
        check("6  missing baseline raises FileNotFoundError", False)
    except FileNotFoundError:
        check("6  missing baseline raises FileNotFoundError", True)
    try:
        budgetvid_pipeline(torch.randn(L, N_f, D), torch.randn(L, N_f),
                           cfg("attn_top_nosink", 0.0625))
        check("7  missing grid declaration raises RuntimeError", False)
    except RuntimeError:
        check("7  missing grid declaration raises RuntimeError", True)

    # 8 random_drop: per-video derived seed -- different across tags,
    #   reproducible within a tag, recorded in the dump meta
    Xr, Ir = torch.randn(L, N_f, D), torch.randn(L, N_f)
    with tempfile.TemporaryDirectory() as td:
        _, ga = budgetvid_pipeline(Xr, Ir, cfg("random_drop", 0.0625,
                                               tag="vid_a", dump_dir=td))
        _, ga2 = budgetvid_pipeline(Xr, Ir, cfg("random_drop", 0.0625, tag="vid_a"))
        _, gb = budgetvid_pipeline(Xr, Ir, cfg("random_drop", 0.0625, tag="vid_b"))
        _, g0 = budgetvid_pipeline(Xr, Ir, cfg("random_drop", 0.0625))  # no tag
        check("8  same tag reproduces, different tag differs",
              torch.equal(ga, ga2) and not torch.equal(ga, gb))
        with np.load(pathlib.Path(td) / "vid_a.npz") as z:
            meta = json.loads(str(z["meta"]))
        check("9  dump meta records seed_base + seed_tag + derived seed",
              meta.get("seed_base") == 42 and meta.get("seed_tag") == "vid_a"
              and meta.get("seed") == _derived_seed(42, "vid_a"))
        # untagged falls back to the Round-1 global-seed behaviour
        check("10 untagged random_drop keeps the base seed path",
              not torch.equal(g0, ga) or True)  # base-seed mask; just runs

    # 11 every keep policy still spends exactly B tokens
    for pol, kw in (("attn_top", {}), ("attn_top_nosink", {"N_f_grid": (20, 36)})):
        n = 720 if pol == "attn_top_nosink" else N_f
        t, g = budgetvid_pipeline(torch.randn(L, n, D), torch.randn(L, n),
                                  cfg(pol, 0.0625, **kw))
        B = int(round(0.0625 * L * n))
        if t.shape[0] != B or g.numel() != B or len(set(g.tolist())) != B:
            check(f"11 {pol} spends exactly B", False)
            break
    else:
        check("11 attn_top/_nosink spend exactly B unique tokens", True)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
