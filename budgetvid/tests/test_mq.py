"""BudgetVID 2.0 core tests: metric lift, CBA curves, water-filling, MPQ, mass.

No GPU, no model, no decord:

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_mq.py

Each test checks a claim the frozen spec makes
(notes/2026-08-28_method_budgetvid2_v1.html), so a failure here is a spec
violation rather than a style complaint.
"""

import itertools
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.core.budget import split_budget                      # noqa: E402
from budgetvid.core.quantize import (                               # noqa: E402
    _argmax_first, _argmin_first, compress_video, fps_curves,
    lower_convex_envelope, metric_lift, quantize_frames, waterfill,
)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def main():
    torch.manual_seed(0)
    L, N_f, d = 8, 24, 16

    # ---- deterministic extrema ------------------------------------------
    v = torch.tensor([[1.0, 3.0, 3.0, 0.0], [5.0, 5.0, 5.0, 5.0]])
    check("01 argmax ties go to the smallest index",
          _argmax_first(v).tolist() == [1, 0])
    check("02 argmin ties go to the smallest index",
          _argmin_first(v).tolist() == [3, 0])

    # ---- metric lift (eq 1) ---------------------------------------------
    x = torch.randn(L, N_f, d)
    W_k = torch.randn(12, d) / d ** 0.5
    W_v = torch.randn(12, d) / d ** 0.5
    g = torch.rand(d) + 0.5
    K, s = metric_lift(x, W_k, W_v, g, gamma_v=1.0)
    rn = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt() * g
    ref = torch.cat([rn @ W_k.t(), rn @ W_v.t()], -1)
    check("03 lift equals W [k|v] . RMSNorm(x)", torch.allclose(K, ref, atol=1e-4),
          f"max |d| = {(K - ref).abs().max():.2e}")
    check("04 s is the scale RMSNorm divides by",
          torch.allclose((x * s).pow(2).mean(-1).sqrt(), torch.ones(L, N_f), atol=1e-3))

    # ---- L1': the norm-weighted centroid is read as the metric centroid --
    grp = x[0, :5]
    sg = s[0, :5]
    ybar = (grp * sg).mean(0)                       # metric-space centroid
    A = lambda z: torch.cat([(z * g) @ W_k.t(), (z * g) @ W_v.t()], -1)
    check("05 L1' A(y_bar) == mean of the lifted group",
          torch.allclose(A(ybar), K[0, :5].mean(0), atol=1e-4),
          f"max |d| = {(A(ybar) - K[0, :5].mean(0)).abs().max():.2e}")
    delivered = ybar / ybar.norm() * grp.norm(dim=-1).mean()
    lifted = metric_lift(delivered[None, None], W_k, W_v, g)[0][0, 0]
    check("05b the delivered token is read as that centroid, direction-exact",
          float(torch.nn.functional.cosine_similarity(
              lifted, K[0, :5].mean(0), dim=0)) > 1 - 1e-6)

    # ---- FPS curves (eq 2) ----------------------------------------------
    seeds, D, r = fps_curves(K, N_f)
    check("06 seeds are a permutation prefix (no repeats)",
          all(len(set(seeds[t].tolist())) == N_f for t in range(L)))
    check("07 D is non-increasing in b", bool((D[:, 1:] <= D[:, :-1] + 1e-4).all()))
    check("08 r is non-increasing in b", bool((r[:, 1:] <= r[:, :-1] + 1e-4).all()))
    check("09 cost and radius vanish at b = N_f",
          float(D[:, -1].max()) < 1e-3 and float(r[:, -1].max()) < 1e-3)

    # duplicates: a frame of 3 distinct tokens repeated 8x saturates at b=3
    dup = x[0, :3].repeat(8, 1).unsqueeze(0)
    Kd, _ = metric_lift(dup, W_k, W_v, g)
    _, Dd, rd = fps_curves(Kd, N_f)
    check("10 a duplicated frame saturates at its distinct count",
          float(rd[0, 2]) < 1e-4 and float(rd[0, 1]) > 1e-4)

    # A saturated frame under the EVEN split is where seeds used to repeat: the
    # residual is all zeros, so "farthest point" has nothing to say, and two
    # seeds landing on one index means two output tokens claiming one position.
    # Water-filling's saturation exit hides it; the ablation path does not.
    dup3 = x[0, :3].repeat(8, 1).unsqueeze(0).repeat(4, 1, 1)      # 4 frames x 24
    _s, _, _ = fps_curves(metric_lift(dup3, W_k, W_v, g)[0], N_f)
    check("10b seeds stay unique even after the frame saturates",
          all(len(set(_s[t].tolist())) == _s.shape[1] for t in range(4)))
    sat = compress_video(dup3, B=40, W_k=W_k, W_v=W_v, g=g, alloc="even")
    idx = [set((t * N_f + si).tolist()) for t, si in enumerate(sat["seed_idx"])]
    check("10c the even split survives a saturated frame",
          sum(len(i) for i in idx) == 40 and len(set().union(*idx)) == 40)

    # ---- envelope --------------------------------------------------------
    Db = lower_convex_envelope(D)
    check("11 envelope lies at or below the curve", bool((Db <= D + 1e-3).all()))
    gains = Db[:, :-1] - Db[:, 1:]
    check("12 envelope gains are non-increasing (convex)",
          bool((gains[:, 1:] <= gains[:, :-1] + 1e-3).all()))
    conv = torch.tensor([[10.0, 6.0, 4.0, 3.0, 2.5]])
    check("13 an already-convex curve is its own envelope",
          torch.allclose(lower_convex_envelope(conv), conv, atol=1e-4))

    # ---- water-filling (eq 3, L2) ---------------------------------------
    B = 3 * L
    b = waterfill(Db, B, r=r)
    check("14 allocation spends exactly B", int(b.sum()) == B, f"{int(b.sum())} vs {B}")
    check("15 every frame keeps the floor of 1", int(b.min()) >= 1)

    # brute force on a small instance: greedy must BE the argmin
    small = lower_convex_envelope(torch.tensor(
        [[9.0, 5.0, 3.0, 2.0], [8.0, 7.5, 7.0, 6.0], [6.0, 2.0, 1.0, 0.5]]))
    bs = waterfill(small, 7)
    best, bestcost = None, float("inf")
    for cand in itertools.product(range(1, 5), repeat=3):
        if sum(cand) != 7:
            continue
        c = sum(float(small[t, cand[t] - 1]) for t in range(3))
        if c < bestcost:
            best, bestcost = cand, c
    got = sum(float(small[t, int(bs[t]) - 1]) for t in range(3))
    check("16 greedy allocation equals the brute-force argmin",
          abs(got - bestcost) < 1e-6, f"greedy {got:.4f} vs best {bestcost:.4f} {best}")

    # a saturated frame must not be handed budget it cannot use
    Ksat = torch.cat([Kd, K[1:4]], 0)
    _, Ds, rs = fps_curves(Ksat, N_f)
    bsat = waterfill(lower_convex_envelope(Ds), 40, r=rs)
    check("17 a saturated frame stops at its distinct count",
          int(bsat[0]) <= 3 and int(bsat.sum()) == 40, f"b0={int(bsat[0])}, sum={int(bsat.sum())}")

    # ---- MPQ (eq 4) ------------------------------------------------------
    feats, sid, mass, cost = quantize_frames(K, x, s, seeds, b)
    check("18 masses sum to N_f in every frame",
          all(int(m.sum()) == N_f for m in mass))
    check("19 every group is non-empty (m_j >= 1)",
          all(int(m.min()) >= 1 for m in mass))
    check("20 one token delivered per seed",
          all(f.shape[0] == int(b[t]) == len(sid[t]) for t, f in enumerate(feats)))
    fb, _, mb, _ = quantize_frames(K, x, s, seeds, torch.full((L,), N_f))
    check("21 at b = N_f the quantizer is the identity",
          torch.allclose(fb[0][seeds[0].argsort()], x[0], atol=1e-4)
          and int(mb[0].max()) == 1,
          f"max |d| = {(fb[0][seeds[0].argsort()] - x[0]).abs().max():.2e}")
    check("21b a merged token keeps a token's magnitude",
          abs(float(feats[0].norm(dim=-1).mean()) - float(x[0].norm(dim=-1).mean()))
          < 0.15 * float(x[0].norm(dim=-1).mean()))

    # ---- L5: the mass bias is an identity, not an approximation ----------
    torch.manual_seed(1)
    n, k, hd = 40, 6, 8
    keys, vals = torch.randn(n, hd), torch.randn(n, hd)
    a = torch.randint(0, k, (n,)); a[:k] = torch.arange(k)
    m = torch.bincount(a, minlength=k).float()
    kk = torch.stack([keys[a == j][0] for j in range(k)])   # identical within group
    keys = kk[a]
    vv = torch.stack([vals[a == j].mean(0) for j in range(k)])
    vals = vv[a]
    q = torch.randn(hd)
    full = torch.softmax(q @ keys.t(), -1) @ vals
    biased = torch.softmax(q @ kk.t() + m.log(), -1) @ vv
    check("22 L5 log-mass bias reproduces the uncompressed read",
          torch.allclose(full, biased, atol=1e-5), f"max |d| = {(full - biased).abs().max():.2e}")
    massless = torch.softmax(q @ kk.t(), -1) @ vv
    check("23 dropping the mass changes the answer (the ablation is not a no-op)",
          not torch.allclose(full, massless, atol=1e-3))

    # ---- end to end ------------------------------------------------------
    out = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g)
    check("24 end to end delivers exactly B tokens",
          sum(f.shape[0] for f in out["feats"]) == 3 * L)
    out2 = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g)
    check("25 end to end is deterministic",
          all(torch.equal(a_, b_) for a_, b_ in zip(out["feats"], out2["feats"])))
    check("26 realized cost is at or above the planned envelope",
          out["cost"] >= out["planned"] - 1e-3,
          f"cost {out['cost']:.3f} planned {out['planned']:.3f}")
    even = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g, alloc="even")
    check("27 alloc=even reproduces the v0 largest-remainder split",
          torch.equal(even["b"].cpu(), split_budget(3 * L, L, N_f)))
    check("28 water-filling costs no more than the even split",
          out["cost"] <= even["cost"] + 1e-6,
          f"waterfill {out['cost']:.3f} vs even {even['cost']:.3f}")
    plain = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g, centroid="plain")
    check("29 centroid=plain is a different delivery (ablation wired)",
          not torch.allclose(plain["feats"][0], out["feats"][0], atol=1e-5))
    # ---- Lloyd refinement (off by default; the frozen v1 path is untouched) --
    ref = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g, refine=5)
    check("29b refine=0 is exactly the frozen behaviour",
          all(torch.equal(a_, b_) for a_, b_ in
              zip(compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g, refine=0)["feats"],
                  out["feats"])))
    check("29c refinement lowers the cost it is supposed to lower",
          ref["cost"] < out["cost"], f"{ref['cost']:.3f} vs {out['cost']:.3f}")
    # The imbalance claim is about clouds with a dense bulk and a sparse tail,
    # which is what real frames look like and what FPS mishandles: it seeds the
    # outliers, and one seed then absorbs the whole bulk. On i.i.d. noise there
    # is no bulk to absorb and nothing to fix, so the claim is tested on the
    # structure it is about.
    bulk = torch.randn(1, 200, d) * 0.05
    tail = torch.randn(1, 24, d) * 4.0
    struct = torch.cat([bulk, tail], 1)
    f0 = compress_video(struct, B=8, W_k=W_k, W_v=W_v, g=g)
    f1 = compress_video(struct, B=8, W_k=W_k, W_v=W_v, g=g, refine=8)
    m0, m1 = int(f0["mass"][0].max()), int(f1["mass"][0].max())
    check("29d on a bulk-plus-outliers cloud, refinement breaks up the mega-group",
          m1 < m0, f"biggest group {m1} vs {m0} of {struct.shape[1]} tokens")
    check("29d2 and it costs less to do so",
          f1["cost"] < f0["cost"], f"{f1['cost']:.2f} vs {f0['cost']:.2f}")
    check("29e refinement keeps every invariant",
          sum(f.shape[0] for f in ref["feats"]) == 3 * L
          and all(int(m_.sum()) == N_f for m_ in ref["mass"])
          and all(int(m_.min()) >= 1 for m_ in ref["mass"]))
    check("29f refinement is deterministic",
          all(torch.equal(a_, b_) for a_, b_ in zip(
              ref["feats"], compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g,
                                           refine=5)["feats"])))
    # Smoke for the refinement's invariants on awkward geometry. It is NOT a
    # regression test for the production failure: several constructions
    # (bulk-plus-outliers, tight clumps, more centers than clumps) all stayed
    # clean under the pre-fix code, so the exact trigger was never reproduced.
    # The guarantee now comes from construction plus an assert inside
    # lloyd_refine, not from this case.
    # 24 tight clumps, every token distinct, budget well under the distinct
    # count -- the regime the failing run was in. Tight clumps are what push two
    # centers on top of each other, which is what made them name one token.
    clump = torch.cat([torch.randn(1, 3, d) * 0.01 + torch.randn(1, 1, d)
                       for _ in range(24)], 1)                     # 72 tokens
    for it in (1, 3, 8):
        rr = compress_video(clump, B=20, W_k=W_k, W_v=W_v, g=g, refine=it)
        si, mm = rr["seed_idx"][0], rr["mass"][0]
        ok = (si.numel() == len(set(si.tolist())) == int(rr["b"].sum()) == 20
              and int(mm.sum()) == clump.shape[1] and int(mm.min()) >= 1)
        check(f"29g0 refine={it} on clumped data: unique positions, no empty group", ok,
              f"{si.numel()} positions, {len(set(si.tolist()))} unique, "
              f"mass {int(mm.sum())}/{clump.shape[1]} min {int(mm.min())}")

    check("29g delivered positions stay real token positions",
          all(int(si.max()) < N_f and int(si.min()) >= 0 for si in ref["seed_idx"])
          and all(len(set(si.tolist())) == si.numel() for si in ref["seed_idx"]))

    # ---- centroid=medoid: a coreset of real tokens, still mass-weighted -----
    med = compress_video(x, B=3 * L, W_k=W_k, W_v=W_v, g=g, centroid="medoid")
    check("29h every delivered token is an actual input token",
          all(any(torch.allclose(f, x[t, i], atol=1e-5) for i in range(N_f))
              for t, F in enumerate(med["feats"]) for f in F))
    check("29i medoid delivery keeps the mass and budget invariants",
          sum(f.shape[0] for f in med["feats"]) == 3 * L
          and all(int(m_.sum()) == N_f for m_ in med["mass"]))
    check("29j a medoid represents its own group, so positions stay unique",
          all(len(set(si.tolist())) == si.numel() for si in med["seed_idx"]))
    # The cost here is a sum of UNSQUARED distances, whose minimizer is the
    # geometric median, not the mean -- so a medoid may well cost less than the
    # centroid, and on this cloud it does. What is guaranteed is the classic
    # metric k-median bound: swapping the mean for the best real point at most
    # doubles the cost.
    check("29k medoid stays within the factor-2 k-median bound",
          med["cost"] <= 2 * out["cost"] + 1e-3,
          f"medoid {med['cost']:.2f} vs centroid {out['cost']:.2f}")
    print(f"      (note: medoid cost {med['cost']:.1f} vs centroid {out['cost']:.1f} "
          f"-- the medoid is the cheaper representative under the L1-type cost)")

    nolift = compress_video(x, B=3 * L)
    check("30 the no-lift ablation runs and still spends B",
          sum(f.shape[0] for f in nolift["feats"]) == 3 * L)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
