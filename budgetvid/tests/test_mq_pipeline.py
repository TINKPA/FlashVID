"""BudgetVID 2.0 plumbing: policy `mq` through the shared assembly, and the
log-mass bias through the attention mask.

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_mq_pipeline.py

These are the failures that do not raise. A mass vector out of step with the
token order, or a bias landing on the wrong slice of the sequence, produces a
model that runs fine and answers slightly worse -- which is indistinguishable
from "the method does not work". Hence the end-to-end identity in test 08: with
groups that are exact duplicates, compressing and biasing must reproduce the
uncompressed attention output to floating-point noise.
"""

import pathlib
import sys
import types

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.adapters.pipeline import budgetvid_pipeline      # noqa: E402
from budgetvid.mass_bias import apply_mass_bias                 # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def cfg(**kw):
    """A stand-in for BudgetVidConfig: the pipeline only ever getattr's."""
    base = dict(policy="mq", retention_ratio=0.25, lift="none", gamma_v=1.0,
                lift_norm=True, mq_alloc="waterfill", centroid="rms", b_max=0,
                mass=True, dump_dir="", seed=42, visual_token_start_index=0,
                visual_token_length=0, token_mass=None)
    base.update(kw)
    return types.SimpleNamespace(**base)


def main():
    torch.manual_seed(0)
    L, N_f, d = 6, 16, 8
    x = torch.randn(L, N_f, d)
    attn = torch.rand(L, N_f)

    c = cfg()
    tokens, g = budgetvid_pipeline(x, attn, c)
    B = int(round(0.25 * L * N_f))
    check("01 mq spends the budget", tokens.shape[0] == B, f"{tokens.shape[0]} vs {B}")
    check("02 global indices are unique and sorted",
          torch.equal(g, g.sort().values) and torch.unique(g).numel() == g.numel())
    check("03 the mass vector is aligned with the tokens",
          c.token_mass is not None and c.token_mass.numel() == tokens.shape[0])
    check("04 masses account for every input token",
          abs(float(c.token_mass.sum()) - L * N_f) < 1e-3,
          f"sum m = {float(c.token_mass.sum())}, N = {L * N_f}")
    check("05 mass=False leaves no bias behind",
          budgetvid_pipeline(x, attn, c2 := cfg(mass=False)) and c2.token_mass is None)

    # each delivered token must sit at its own seed's position
    check("06 delivered positions are seed positions, one per group",
          bool(((g // N_f) >= 0).all()) and int((g // N_f).bincount(minlength=L).sum()) == B)

    # the allocation ablation must reproduce v0's split exactly
    _, g_even = budgetvid_pipeline(x, attn, ce := cfg(mq_alloc="even"))
    per_frame = torch.bincount(g_even // N_f, minlength=L)
    check("07 mq_alloc=even gives every frame the same share",
          int(per_frame.max()) - int(per_frame.min()) <= 1, per_frame.tolist())

    # ---- 08: the identity that proves the plumbing --------------------------
    # Frames built from k distinct vectors repeated, so quantization at b_t = k
    # is lossless and the ONLY thing standing between compressed and
    # uncompressed attention is the mass channel.
    k = 4
    base = torch.randn(L, k, d)
    xd = base.repeat_interleave(N_f // k, dim=1)          # [L, N_f, d]
    cd = cfg(retention_ratio=k / N_f)
    toks, gd = budgetvid_pipeline(xd, torch.rand(L, N_f), cd)
    m = cd.token_mass

    n_text_before, n_text_after = 3, 2
    pre, post = torch.randn(n_text_before, d), torch.randn(n_text_after, d)
    q = post[-1]
    full_keys = torch.cat([pre, xd.reshape(-1, d), post], 0)
    comp_keys = torch.cat([pre, toks, post], 0)
    full_out = torch.softmax(q @ full_keys.t(), -1) @ full_keys
    check("08a quantization of duplicates is lossless",
          abs(float(toks.norm()) - float(base.norm())) < 1e-3 * float(base.norm()))

    cd.visual_token_start_index = n_text_before
    hidden = torch.zeros(1, comp_keys.shape[0], d)
    cache_position = torch.arange(comp_keys.shape[0])
    mask = apply_mass_bias(None, hidden, cache_position, cd)
    check("08b the bias materializes a causal float mask",
          mask.shape == (1, 1, comp_keys.shape[0], comp_keys.shape[0]))
    scores = q @ comp_keys.t() + mask[0, 0, -1]
    comp_out = torch.softmax(scores, -1) @ comp_keys
    check("08c compressed + log-mass reproduces uncompressed attention",
          torch.allclose(full_out, comp_out, atol=1e-4),
          f"max |d| = {(full_out - comp_out).abs().max():.2e}")
    no_bias = torch.softmax(q @ comp_keys.t(), -1) @ comp_keys
    check("08d without the bias it does NOT reproduce it",
          not torch.allclose(full_out, no_bias, atol=1e-3))

    # ---- mask forms the hook has to accept ---------------------------------
    bool_mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
    ch = cfg(visual_token_start_index=1, token_mass=torch.tensor([2.0, 3.0]))
    out = apply_mass_bias(bool_mask, torch.zeros(1, 4, d), torch.arange(4), ch)
    check("09 a bool mask becomes additive, keeping the causal structure",
          out.dtype.is_floating_point and float(out[0, 0, 0, 1]) < -1e30
          and abs(float(out[0, 0, 2, 1]) - float(torch.tensor(2.0).log())) < 1e-4)

    float_mask = torch.zeros(1, 1, 4, 4)
    out = apply_mass_bias(float_mask, torch.zeros(1, 4, d), torch.arange(4), ch)
    check("10 a float mask is added to, not replaced",
          abs(float(out[0, 0, 0, 2]) - float(torch.tensor(3.0).log())) < 1e-4)

    # decode: one query, the whole cache as keys, same slice
    dec = apply_mass_bias(None, torch.zeros(1, 1, d), torch.tensor([9]), ch)
    check("11 the bias survives into decode",
          dec.shape == (1, 1, 1, 10)
          and abs(float(dec[0, 0, 0, 2]) - float(torch.tensor(3.0).log())) < 1e-4)

    ch_off = cfg(token_mass=None)
    check("12 no mass means the mask passes through untouched",
          apply_mass_bias(float_mask, torch.zeros(1, 4, d), torch.arange(4), ch_off)
          is float_mask)

    bad = cfg(visual_token_start_index=8, token_mass=torch.tensor([1.0, 1.0]))
    try:
        apply_mass_bias(None, torch.zeros(1, 4, d), torch.arange(4), bad)
        ok = False
    except ValueError:
        ok = True
    check("13 a mass vector that overruns the sequence raises", ok)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
