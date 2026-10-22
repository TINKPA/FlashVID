"""2026-10-21 iteration knobs: protected singleton channel + mass-bias shaping.

No GPU, no model, no decord:

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_protect_massform.py

Covers three claims the campaign design makes
(notes/2026-10-21_experiment_protect-massform.html):
  P: `protect` delivers the globally most isolated tokens verbatim as m=1
     groups, spends exactly B, and with protect=0 the path is bit-identical
     to the frozen one.
  C: `mass_cap` caps only what the bias sees; the true mass is untouched.
  E: `mass_eff` keeps log m for pure groups and decays it for scattered ones,
     and survives a FastV-style re-index in lockstep with the true mass.
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.core.quantize import (                               # noqa: E402
    compress_video, fps_curves, isolation_scores, quantize_frames, metric_lift,
)
from budgetvid.mass_bias import bias_mass                           # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


class _Cfg:
    """The three attributes bias_mass reads, nothing else."""

    def __init__(self, m, me=None, cap=0.0):
        self.token_mass = m
        self.token_mass_eff = me
        self.mass_cap = cap


def main():
    torch.manual_seed(0)
    L, N_f, d = 6, 40, 16

    # A video with planted outliers: token 7 of every frame sits far away.
    x = torch.randn(L, N_f, d) * 0.1
    x[:, 7] += 8.0

    # ---- isolation ------------------------------------------------------
    K, _ = metric_lift(x, None, None, None, 1.0, 1e-6)
    nn = isolation_scores(K)
    check("P01 planted outliers have the top isolation score in every frame",
          bool((nn.argmax(dim=1) == 7).all()), str(nn.argmax(dim=1)))
    xdup = x.clone(); xdup[0, 1] = xdup[0, 0]
    nndup = isolation_scores(metric_lift(xdup, None, None, None, 1.0, 1e-6)[0])
    check("P02 a duplicated token scores 0 and can never be protected",
          float(nndup[0, 0]) == 0.0 and float(nndup[0, 1]) == 0.0)

    # ---- fps_curves exclusion -------------------------------------------
    excl = torch.zeros(L, N_f, dtype=torch.bool); excl[:, 7] = True
    seeds_x, D_x, r_x = fps_curves(K, 12, exclude=excl)
    check("P03 excluded tokens never appear among the used seeds",
          not bool((seeds_x == 7).any()))
    seeds_0, D_0, r_0 = fps_curves(K, 12)
    check("P04 exclude=None reproduces the frozen curves bit for bit",
          torch.equal(*[t for t in [fps_curves(K, 12, exclude=None)[1], D_0]]))
    bad = torch.ones(L, N_f, dtype=torch.bool)
    try:
        fps_curves(K, 4, exclude=bad); check("P05 a fully excluded frame raises", False)
    except ValueError:
        check("P05 a fully excluded frame raises", True)

    # ---- compress_video with protect ------------------------------------
    B = 60
    out0 = compress_video(x, B)
    outp = compress_video(x, B, protect=0.2)
    n_p = int(round(0.2 * B))
    check("P06 protect reports round(protect*B) protected tokens",
          outp["n_protected"] == n_p, str(outp.get("n_protected")))
    check("P07 budget is spent exactly: sum(b) == B",
          int(outp["b"].sum()) == B, str(int(outp["b"].sum())))
    masses = torch.cat(outp["mass"])
    check("P08 total mass is conserved: sum(m) == L*N_f",
          int(masses.sum()) == L * N_f, str(int(masses.sum())))
    # the planted outliers must be protected, delivered verbatim, with m=1
    ok_feat, ok_mass, found = True, True, 0
    for t in range(L):
        sid = outp["seed_idx"][t]
        for j, sidx in enumerate(sid.tolist()):
            if bool(outp["protected"][t, sidx]):
                found += 1
                ok_mass &= int(outp["mass"][t][j]) == 1
                ok_feat &= torch.allclose(outp["feats"][t][j], x[t, sidx].float(),
                                          atol=1e-5)
    check("P09 every protected token is delivered (count matches)", found == n_p,
          f"{found} vs {n_p}")
    check("P10 protected groups have m == 1", ok_mass)
    check("P11 protected tokens are delivered verbatim (identity)", ok_feat)
    check("P12 planted outliers are among the protected",
          bool(outp["protected"][:, 7].sum() >= L - 1), str(int(outp["protected"][:, 7].sum())))
    check("P13 protect=0 is bit-identical to the frozen path (b, cost)",
          torch.equal(out0["b"], compress_video(x, B, protect=0.0)["b"])
          and out0["cost"] == compress_video(x, B, protect=0.0)["cost"])
    check("P14 group_of maps protected tokens to themselves",
          all(int(outp["group_of"][t, i]) == t * N_f + i
              for t in range(L) for i in torch.nonzero(outp["protected"][t]).flatten().tolist()))
    try:
        compress_video(x, L + 1, protect=0.9)
        check("P15 protect that starves the floor raises", False)
    except ValueError:
        check("P15 protect that starves the floor raises", True)

    # ---- mass_eff --------------------------------------------------------
    # frame 0: two tight planted clusters -> both groups pure -> m_eff == m
    xp = torch.zeros(2, 8, d)
    xp[0, :4, 0] = 1.0; xp[0, 4:, 1] = 50.0
    xp[1, :, 2] = torch.arange(8).float() * 3.0      # a spread frame
    Kp, sp = metric_lift(xp, None, None, None, 1.0, 1e-6)
    me = []
    seeds_p, _, _ = fps_curves(Kp, 8)
    b_p = torch.tensor([2, 2])
    feats, sid, mass, cost = quantize_frames(Kp, xp, sp, seeds_p, b_p,
                                             mass_eff_out=me)
    pure = torch.allclose(me[0].sort().values, mass[0].float().sort().values, atol=1e-4)
    check("E01 pure groups keep m_eff == m", pure, f"{me[0]} vs {mass[0]}")
    check("E02 scattered groups decay: m_eff < m (strictly, where m > 1)",
          bool((me[1][mass[1] > 1] < mass[1][mass[1] > 1].float()).all()),
          f"{me[1]} vs {mass[1]}")
    check("E03 m_eff >= 1 everywhere", bool((me[0] >= 1).all() and (me[1] >= 1).all()))
    outm = compress_video(x, B, mass_eff=True)
    check("E04 compress_video(mass_eff) returns per-frame vectors matching mass shape",
          all(outm["mass_eff"][t].shape == outm["mass"][t].shape for t in range(L)))
    check("E05 mass_eff <= m elementwise",
          all(bool((outm["mass_eff"][t] <= outm["mass"][t].float() + 1e-4).all())
              for t in range(L)))

    # ---- bias shaping ----------------------------------------------------
    m = torch.tensor([1.0, 4.0, 32.0])
    check("C01 cap=0 leaves the bias at log m",
          torch.allclose(bias_mass(_Cfg(m)), m.log()))
    check("C02 cap=8 caps only the big group",
          torch.allclose(bias_mass(_Cfg(m, cap=8.0)),
                         torch.tensor([1.0, 4.0, 8.0]).log()))
    me_v = torch.tensor([1.0, 1.5, 2.0])
    check("C03 token_mass_eff replaces m in the bias",
          torch.allclose(bias_mass(_Cfg(m, me=me_v)), me_v.log()))
    check("C04 cap applies after eff",
          torch.allclose(bias_mass(_Cfg(m, me=torch.tensor([1.0, 1.5, 12.0]), cap=8.0)),
                         torch.tensor([1.0, 1.5, 8.0]).log()))
    # FastV-style re-index keeps the two vectors in lockstep
    ranks = torch.tensor([2, 0])
    cfg = _Cfg(m, me=me_v)
    cfg.token_mass = cfg.token_mass[ranks]; cfg.token_mass_eff = cfg.token_mass_eff[ranks]
    check("C05 re-indexed eff bias follows the same ranks",
          torch.allclose(bias_mass(cfg), me_v[ranks].log()))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
