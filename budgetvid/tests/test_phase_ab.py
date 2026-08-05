"""The ten Phase A/B tests the plan requires. No GPU, no model, no decord.

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_phase_ab.py
"""

import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.core.assembly import assemble          # noqa: E402
from budgetvid.core.budget import split_budget        # noqa: E402
from budgetvid.core.merging import seeded_merge       # noqa: E402
from budgetvid.core.routing import beta_window, route_tokens  # noqa: E402
from budgetvid.core.scoring import score_tokens       # noqa: E402

L, N_F, D, GRID = 32, 196, 32, (14, 14)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}{'  ' + detail if detail else ''}")


def fixture(seed=0):
    torch.manual_seed(seed)
    X = torch.randn(L, N_F, D)
    I = torch.rand(L, N_F)
    return X, I, score_tokens(X, I, GRID)


def main() -> int:
    X, I, sc = fixture()
    b = split_budget(627, L, N_F)

    # 1 budget identity per frame
    rt = route_tokens(sc["S"], sc["R_raw_mean"], b, beta=4.0)
    check("1  |retain| + B_M == b_t, every frame",
          all(len(rt["retain_idx"][t]) + int(rt["B_M"][t]) == int(b[t]) for t in range(L)))

    # 2 total budget identity, including the non-divisible case
    ok = True
    for B in (627, 1568, 100, 641):
        try:
            bb = split_budget(B, L, N_F)
            ok &= int(bb.sum()) == B
        except ValueError:
            ok = False
    check("2  sum(b_t) == B, incl. L not dividing B", ok,
          f"627 -> {sorted(set(split_budget(627, L, N_F).tolist()))}")

    # 3 feasibility invariant
    inv = all(int(rt["N_active"][t]) >= int(b[t]) and
              int(rt["N_active"][t]) - int(rt["B_R"][t]) >= int(rt["B_M"][t])
              for t in range(L))
    check("3  N_active >= b_t  and  |pool| >= B_M", inv)

    # 4 rank normalisation: every frame's R_hat averages to 0.5
    check("4  per-frame mean(R_hat) == 0.5",
          torch.allclose(sc["R_hat"].mean(-1), torch.full((L,), 0.5), atol=1e-6))

    # 5 cross-frame percentile ordering: a more redundant frame gets lower alpha
    Xs = torch.randn(4, N_F, D)
    Xs[0] = Xs[0, :1].repeat(N_F, 1)          # frame 0 = maximally homogeneous
    Xs[3] = torch.randn(N_F, D) * 5           # frame 3 = maximally varied
    sc2 = score_tokens(Xs, torch.rand(4, N_F), GRID)
    rt2 = route_tokens(sc2["S"], sc2["R_raw_mean"], split_budget(80, 4, N_F), beta=4.0)
    hi = int(sc2["R_raw_mean"].argmax())
    lo = int(sc2["R_raw_mean"].argmin())
    check("5  alpha_t lower for the more redundant frame",
          float(rt2["alpha"][hi]) < float(rt2["alpha"][lo]),
          f"alpha[most redundant]={float(rt2['alpha'][hi]):.2f} < alpha[least]={float(rt2['alpha'][lo]):.2f}")

    # 6 the three sets partition [N_f] exactly
    part = all(
        sorted(torch.cat([rt["retain_idx"][t], rt["pool_idx"][t], rt["drop_idx"][t]]).tolist())
        == list(range(N_F)) for t in range(L))
    check("6  retain | pool | drop is a partition of [N_f]", part)

    # 7 determinism
    _, _, scb = fixture()
    rtb = route_tokens(scb["S"], scb["R_raw_mean"], b, beta=4.0)
    check("7  identical inputs -> identical routing",
          all(torch.equal(rt["retain_idx"][t], rtb["retain_idx"][t]) for t in range(L))
          and torch.equal(sc["S"], scb["S"]))

    # 8 alpha flip is a mirror about (alpha_min+alpha_max)/2
    rtf = route_tokens(sc["S"], sc["R_raw_mean"], b, beta=4.0, alpha_flip=True)
    check("8  alpha_flip mirrors alpha about the midpoint",
          torch.allclose(rt["alpha"] + rtf["alpha"], torch.full((L,), 1.2), atol=1e-6))

    # 9 force_alpha=1 -> pure pruning, empty merge pool  (Step 4's precondition)
    rt1 = route_tokens(sc["S"], sc["R_raw_mean"], b, beta=4.0, force_alpha=1.0)
    check("9  force_alpha=1 => B_R=b_t, B_M=0, pool empty",
          torch.equal(rt1["B_R"], b) and int(rt1["B_M"].sum()) == 0
          and all(p.numel() == 0 for p in rt1["pool_idx"]))

    # 10 beta window labelling
    r10, r25 = 627 / (L * N_F), 1568 / (L * N_F)
    check("10 beta window labels match the effective range",
          beta_window(2, r10) == "below" and beta_window(8, r10) == "in"
          and beta_window(2, r25) == "in" and beta_window(4, r25) == "above",
          f"r=10% beta=2 -> {beta_window(2, r10)}, beta=8 -> {beta_window(8, r10)}")

    # --- extras: Phase C/D, which Step 4 depends on ---
    merged = [seeded_merge(X[t], rt["pool_idx"][t], sc["S"][t], int(rt["B_M"][t]))
              for t in range(L)]
    check("11 seeded_merge emits exactly B_M tokens",
          all(merged[t][0].shape[0] == int(rt["B_M"][t]) for t in range(L)))
    tok, g = assemble(X, rt["retain_idx"], merged=merged, expected_total=627)
    check("12 assemble: exactly B tokens, indices unique and ascending",
          tok.shape[0] == 627 and g.unique().numel() == 627 and bool((g.diff() > 0).all()))

    # identity degeneracy: b_t = N_f and alpha = 1 must return the input verbatim
    bfull = torch.full((L,), N_F, dtype=torch.long)
    rid = route_tokens(sc["S"], sc["R_raw_mean"], bfull, beta=4.0, force_alpha=1.0)
    tid, gid = assemble(X, rid["retain_idx"], merged=None, expected_total=L * N_F)
    check("13 identity degeneracy is bit-exact",
          torch.equal(tid, X.reshape(L * N_F, D)) and torch.equal(gid, torch.arange(L * N_F)))

    # active_frac must not drift with r, unlike beta
    for B, name in ((627, "r=10%"), (78, "r=1.25%")):
        bb = split_budget(B, L, N_F)
        ra = route_tokens(sc["S"], sc["R_raw_mean"], bb, active_frac=0.6)
        frac = float(ra["N_active"].float().mean()) / N_F
        check(f"14 active_frac=0.6 holds at {name}", abs(frac - 0.6) < 0.02, f"got {frac:.3f}")

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
