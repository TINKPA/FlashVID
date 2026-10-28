"""The score-free floors `uniform` and `random_drop` (restored 2026-10-27 for the
baselines x frames grid).

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_floor_policies.py

A floor row is only a floor if it spends the same budget as the method rows,
keeps real tokens (no merge, no mass) and, for random, draws a different mask
per video that still reproduces across runs.
"""

import pathlib
import sys
import types

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.adapters.pipeline import budgetvid_pipeline      # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def cfg(**kw):
    base = dict(policy="uniform", retention_ratio=0.25, dump_dir="", seed=42,
                dump_tag="", visual_token_start_index=0, visual_token_length=0,
                token_mass=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def main():
    torch.manual_seed(0)
    for (L, N_f) in [(16, 299), (64, 299), (128, 120)]:
        x = torch.randn(L, N_f, 4)
        a = torch.rand(L, N_f)
        for r in (0.10, 0.15, 0.20, 0.25):
            B = int(round(r * L * N_f))
            for pol in ("uniform", "random_drop"):
                c = cfg(policy=pol, retention_ratio=r, dump_tag="vid_a")
                tok, g = budgetvid_pipeline(x, a, c)
                tag = f"{pol} L={L} N_f={N_f} r={r}"
                check(f"spends B: {tag}", tok.shape[0] == B == c.visual_token_length,
                      f"{tok.shape[0]} vs {B}")
                check(f"unique sorted: {tag}",
                      torch.equal(g, g.sort().values) and torch.unique(g).numel() == g.numel())
                check(f"original tokens: {tag}",
                      torch.equal(tok, x.reshape(L * N_f, -1)[g]))
                check(f"no mass: {tag}", c.token_mass is None)
                per = torch.bincount(g // N_f, minlength=L)
                check(f"even split: {tag}", int(per.max() - per.min()) <= 1)

    x = torch.randn(16, 299, 4); a = torch.rand(16, 299)
    _, g1 = budgetvid_pipeline(x, a, cfg(policy="random_drop", dump_tag="vid_a"))
    _, g2 = budgetvid_pipeline(x, a, cfg(policy="random_drop", dump_tag="vid_a"))
    _, g3 = budgetvid_pipeline(x, a, cfg(policy="random_drop", dump_tag="vid_b"))
    check("random reproduces for the same video", torch.equal(g1, g2))
    check("random differs across videos", not torch.equal(g1, g3))
    _, u1 = budgetvid_pipeline(x, a, cfg(policy="uniform", dump_tag="vid_a"))
    _, u2 = budgetvid_pipeline(x, a, cfg(policy="uniform", dump_tag="vid_b"))
    check("uniform is video-independent", torch.equal(u1, u2))
    _, v = budgetvid_pipeline(x, a, cfg(policy="none"))
    check("policy none still passes everything", v.numel() == 16 * 299)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
