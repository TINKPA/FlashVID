"""Full-length reconstruction (the `reconstruct` oracle): mapping + pipeline.

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_reconstruct.py

The failure this guards against does not raise: a reconstruction whose rows are
off by one (searchsorted on a value that is not present) or whose indices are
not arange hands the backbone a full-length sequence that silently carries the
wrong features at the right positions, or the right features filtered to the
wrong positions.
"""

import pathlib
import sys
import types

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.adapters.pipeline import budgetvid_pipeline       # noqa: E402
from budgetvid.core.assembly import assemble, reconstruct_full   # noqa: E402
from budgetvid.core.quantize import compress_video               # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def cfg(**kw):
    base = dict(policy="mq", retention_ratio=0.25, lift="none", gamma_v=1.0,
                lift_norm=True, mq_alloc="waterfill", centroid="rms", b_max=0,
                mass=True, dump_dir="", visual_token_start_index=0,
                visual_token_length=0, token_mass=None, reconstruct="none")
    base.update(kw)
    return types.SimpleNamespace(**base)


def main():
    torch.manual_seed(0)
    L, N_f, d = 6, 16, 8
    x = torch.randn(L, N_f, d)
    attn = torch.rand(L, N_f)
    B = int(round(0.25 * L * N_f))

    # ---- the mapping itself, against compress_video's own ground truth -----
    out = compress_video(x, B)
    empty = [torch.empty(0, dtype=torch.long)] * L
    tokens, gidx = assemble(x, empty, merged=list(zip(out["feats"], out["seed_idx"])))
    full = reconstruct_full(tokens, gidx, out["group_of"])
    check("01 reconstruction is full length", full.shape == (L * N_f, tokens.shape[1]))
    ok = True
    flat = out["group_of"].reshape(-1)
    for i in range(L * N_f):
        j = int((gidx == flat[i]).nonzero().item())
        if not torch.equal(full[i], tokens[j]):
            ok = False
            break
    check("02 every position carries its OWN group's delivered token", ok)
    check("03 a delivered position carries the delivered token itself",
          all(torch.equal(full[int(g)], tokens[j]) for j, g in enumerate(gidx)))

    bad = out["group_of"].clone()
    bad[0, 0] = int(gidx[0]) + 1 if int(gidx[0]) + 1 not in set(gidx.tolist()) else int(gidx[0]) - 1
    try:
        reconstruct_full(tokens, gidx, bad)
        raised = False
    except AssertionError:
        raised = True
    check("04 a group_of entry pointing at a never-delivered index raises", raised)

    # ---- through the pipeline ----------------------------------------------
    c = cfg(reconstruct="group")
    toks, g = budgetvid_pipeline(x, attn, c)
    check("05 group mode returns L*N_f tokens with arange indices",
          toks.shape[0] == L * N_f and torch.equal(g, torch.arange(L * N_f)))
    check("06 group mode forces the mass channel off",
          c.token_mass is None and c.visual_token_length == L * N_f)
    check("07 group mode equals the standalone reconstruction",
          torch.allclose(toks, full.to(toks.dtype)))
    n_groups = torch.unique(out["group_of"]).numel()
    check("08 group mode carries exactly B distinct rows",
          torch.unique(toks, dim=0).shape[0] <= B and n_groups == B,
          f"{torch.unique(toks, dim=0).shape[0]} vs {B}")

    ci = cfg(reconstruct="identity")
    toks_i, g_i = budgetvid_pipeline(x, attn, ci)
    check("09 identity mode returns the input bit-exactly",
          torch.equal(toks_i, x.reshape(L * N_f, -1)) and torch.equal(g_i, torch.arange(L * N_f)))
    check("10 identity mode forces the mass channel off", ci.token_mass is None)

    try:
        budgetvid_pipeline(x, attn, cfg(reconstruct="group", llm_prune="fastv"))
        raised = False
    except ValueError:
        raised = True
    check("11 reconstruct with llm_prune=fastv raises", raised)

    c_off = cfg(reconstruct="none")
    toks_off, g_off = budgetvid_pipeline(x, attn, c_off)
    check("12 reconstruct='none' is the frozen compressed path",
          toks_off.shape[0] == B and c_off.token_mass is not None)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        print("FAILED:", FAIL)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
