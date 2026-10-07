"""Method bv's inner-LLM stage (budgetvid/llm_prune.py), no GPU, no transformers.

    uv run --no-project --python 3.11 --with torch --with numpy python budgetvid/tests/test_llm_prune.py
"""
import math
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# flashvid/__init__.py imports transformers; the helper under test only needs
# torch, so register the package as a bare namespace (as test_flashvid_mass does).
if "flashvid" not in sys.modules:
    _pkg = types.ModuleType("flashvid")
    _pkg.__path__ = [str(ROOT / "flashvid")]
    sys.modules["flashvid"] = _pkg

from budgetvid.llm_prune import (  # noqa: E402
    bv_llm_pruning, mass_score_bias, reset_inner_state)

PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    PASS += ok; FAIL += (not ok)
    print(("  ok  " if ok else "  FAIL") + " " + name + ("" if ok or not detail else f"  [{detail}]"))


def cfg(**kw):
    c = types.SimpleNamespace(pruning_layer=20, llm_retention_ratio=0.3, llm_prune="none",
                              visual_token_start_index=0, visual_token_length=0, token_mass=None)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


def sample(n_before, n_vis, n_after, d=8, hot=()):
    L = n_before + n_vis + n_after
    hidden = torch.randn(1, L, d)
    attn = torch.zeros(1, 4, L, L)
    for j in hot:
        attn[:, :, -1, n_before + j] = 1.0 + 0.01 * j
    pos = torch.arange(L)
    emb = (torch.randn(1, L, d), torch.randn(1, L, d))
    return hidden, attn, pos, emb


def main():
    n_before, n_vis, n_after, d = 5, 20, 4, 8
    L = n_before + n_vis + n_after

    print("llm_prune=none:")
    c = cfg(visual_token_start_index=n_before, visual_token_length=n_vis)
    hidden, attn, pos, emb = sample(n_before, n_vis, n_after, d)
    mask = torch.zeros(1, 1, L, L)
    h, m, pid, cp, pe, keep = bv_llm_pruning(hidden, mask, attn, pos, pos.unsqueeze(0), emb, c)
    check("keeps every position", keep.tolist() == list(range(L)))
    check("hidden states untouched", h is hidden)
    check("mask untouched", m is mask)
    check("config length untouched", c.visual_token_length == n_vis)
    check("not recorded as pruned", getattr(c, "inner_pruned", False) is False)

    print("llm_prune=fastv, FlashVID settings (layer 20, keep 0.3):")
    hot = (2, 5, 7, 11, 13, 19)        # the last query attends to these visual tokens
    k = math.ceil(n_vis * 0.3)         # 6
    mass = torch.arange(1, n_vis + 1)
    c = cfg(llm_prune="fastv", visual_token_start_index=n_before, visual_token_length=n_vis,
            token_mass=mass.clone())
    hidden, attn, pos, emb = sample(n_before, n_vis, n_after, d, hot=hot)
    floor = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, L, L)
    mask.masked_fill_(torch.ones(L, L, dtype=torch.bool).triu(1), floor)
    mask[..., n_before:n_before + n_vis] += mass.float().log()
    h, m, pid, cp, pe, keep = bv_llm_pruning(hidden, mask, attn, pos, pos.unsqueeze(0), emb, c)
    kept_vis = sorted(int(i) - n_before for i in keep if n_before <= int(i) < n_before + n_vis)
    check(f"keeps ceil(0.3 * V) = {k} visual tokens", len(kept_vis) == k, str(len(kept_vis)))
    check("they are the ones the last query attended to", kept_vis == sorted(hot), str(kept_vis))
    text = [int(i) for i in keep if not (n_before <= int(i) < n_before + n_vis)]
    check("every text token survives, in order",
          text == list(range(n_before)) + list(range(n_before + n_vis, L)), str(text))
    check("keep_indices sorted", keep.tolist() == sorted(keep.tolist()))
    check("hidden states gathered by keep", torch.equal(h[0], hidden[0, keep]))
    check("rope cos/sin gathered by keep", torch.equal(pe[0][0], emb[0][0, keep]) and torch.equal(pe[1][0], emb[1][0, keep]))
    check("cache_position / position_ids gathered", cp.tolist() == keep.tolist() and pid[0].tolist() == keep.tolist())
    check("config length is the kept count", c.visual_token_length == k)
    check("recorded as pruned", c.inner_pruned is True)
    check("kept ranks recorded for the dumps", c._bv_llm_keep.tolist() == sorted(hot))
    check("mass vector re-indexed to the kept keys", c.token_mass.tolist() == [int(mass[j]) for j in sorted(hot)],
          str(c.token_mass.tolist()))
    new_len = n_before + k + n_after
    check("mask shrinks to the kept sequence", tuple(m.shape) == (1, 1, new_len, new_len), str(tuple(m.shape)))
    last = m[0, 0, -1]
    check("last query: log m of the KEPT tokens on the visual keys",
          torch.allclose(last[n_before:n_before + k], mass[sorted(hot)].float().log()))
    check("last query: no bias on text keys",
          torch.equal(last[:n_before], torch.zeros(n_before)) and torch.equal(last[n_before + k:], torch.zeros(n_after)))
    upper = torch.ones(new_len, new_len, dtype=torch.bool).triu(1)
    check("causal structure preserved", bool((m[0, 0][upper] <= floor / 2).all()) and bool((m[0, 0][~upper] > floor / 2).all()))

    print("score bias around pruning:")
    dec = torch.zeros(1, 1, 1, new_len + 1)
    check("decode after pruning: mask returned untouched",
          mass_score_bias(dec, torch.zeros(1, 1, d), torch.tensor([99]), c) is dec)
    check("decode after pruning: None stays None",
          mass_score_bias(None, torch.zeros(1, 1, d), torch.tensor([99]), c) is None)
    c2 = cfg(visual_token_start_index=n_before, visual_token_length=n_vis, token_mass=mass.clone(), inner_pruned=False)
    cache = types.SimpleNamespace(get_seq_length=lambda layer_idx=0: L)
    out = mass_score_bias(None, torch.zeros(1, 1, d), torch.tensor([99]), c2, cache)
    check("decode without pruning: bias materialized over the cached keys",
          out is not None and tuple(out.shape) == (1, 1, 1, L + 1)
          and torch.allclose(out[0, 0, 0, n_before:n_before + n_vis], mass.float().log()))
    pre = mass_score_bias(None, torch.zeros(1, L, d), torch.arange(L), c2)
    check("prefill: bias on the visual keys", torch.allclose(pre[0, 0, -1, n_before:n_before + n_vis], mass.float().log()))

    print("per-sample reset:")
    reset_inner_state(c)
    check("inner_pruned cleared", c.inner_pruned is False)
    check("kept ranks cleared", c._bv_llm_keep is None)

    print("fastv without the mass channel:")
    c3 = cfg(llm_prune="fastv", visual_token_start_index=n_before, visual_token_length=n_vis, token_mass=None)
    hidden, attn, pos, emb = sample(n_before, n_vis, n_after, d, hot=hot)
    mask = torch.zeros(1, 1, L, L)
    h, m, *_ , keep = bv_llm_pruning(hidden, mask, attn, pos, pos.unsqueeze(0), emb, c3)
    check("mass=False: plain FlashVID output, mask only sliced", tuple(m.shape) == (1, 1, new_len, new_len) and c3.token_mass is None)

    print("unknown mode:")
    try:
        bv_llm_pruning(hidden, mask, attn, pos, pos.unsqueeze(0), emb, cfg(llm_prune="bogus", visual_token_start_index=0, visual_token_length=1))
        check("raises", False)
    except KeyError:
        check("raises KeyError", True)

    print(f"\n{PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
