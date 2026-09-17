"""Method ``fvmass``: FlashVID with the mass channel.

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_flashvid_mass.py

The failures that matter here do not raise. If the instrumented copy of FlashVID's
vision-side compression drifts from the original, the mass=False arm stops being
FlashVID and the plug-in comparison measures the drift instead of the mass. If a
mass lands on the wrong key after inner-LLM pruning, the model runs and answers
slightly differently. So: token-for-token equality with ``flashvid.utils``, the
group sizes against a hand-built tree, the bias position after pruning, and the
decode mask with inner pruning on (untouched) and off (biased).

``flashvid/__init__.py`` imports transformers and llava; the helpers under test only
need torch, so the package is registered as a bare namespace pointing at the source
directory and its ``__init__`` is never executed.
"""

import math
import pathlib
import sys
import types

import torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "flashvid" not in sys.modules:
    _pkg = types.ModuleType("flashvid")
    _pkg.__path__ = [str(ROOT / "flashvid")]
    sys.modules["flashvid"] = _pkg

from flashvid.configuration_flashvid import FlashVidConfig          # noqa: E402
from flashvid.utils import flashvid_compression                     # noqa: E402
from budgetvid.flashvid_mass import (                               # noqa: E402
    _spatiotemporal_compression_with_mass, flashvid_compression_with_mass,
    fvmass_prune, fvmass_score_bias)
from budgetvid.mass_bias import apply_mass_bias                     # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def published_qwen25(**kw):
    """FlashVID's released Qwen2.5-VL settings (its scripts/qwen2_5_vl.sh)."""
    base = dict(retention_ratio=0.10, alpha=0.7, token_selection_method="attn_div",
                temporal_threshold=0.8, do_segment=True, segment_threshold=0.9,
                min_segment_num=4, complementary_segment=True, expansion=1.25,
                pruning_layer=20, llm_retention_ratio=0.3)
    base.update(kw)
    return FlashVidConfig(**base)


def video(seed, F=16, N=64, D=24, drift=0.15, scene_every=5):
    """Frames that drift slowly inside a scene and jump between scenes, so both
    TSTM merges and several segments actually occur."""
    g = torch.Generator().manual_seed(seed)
    frames, cur = [], torch.randn(N, D, generator=g)
    for f in range(F):
        if f % scene_every == 0 and f > 0:
            cur = torch.randn(N, D, generator=g)
        cur = cur + drift * torch.randn(N, D, generator=g)
        frames.append(cur.clone())
    feats = torch.stack(frames)
    cls = torch.rand(F, N, generator=g)
    return feats, cls


def compare(tag, feats, cls, cfg_kw):
    ref_cfg, mass_cfg = published_qwen25(**cfg_kw), published_qwen25(**cfg_kw)
    mass_cfg.mass = True
    mass_cfg.fvmass_pruned = True   # left over from the previous video's prefill
    ref_tok, ref_idx = flashvid_compression(feats.clone(), cls.clone(), ref_cfg)
    tok, idx = flashvid_compression_with_mass(feats.clone(), cls.clone(), mass_cfg)
    check(f"{tag}: compression clears the previous video's pruning record", mass_cfg.fvmass_pruned is False)
    check(f"{tag}: same number of tokens", tok.shape == ref_tok.shape, f"{tuple(tok.shape)} vs {tuple(ref_tok.shape)}")
    if tok.shape == ref_tok.shape:
        check(f"{tag}: tokens bit-identical", torch.equal(tok, ref_tok),
              f"max |diff| {float((tok - ref_tok).abs().max()):.3g}")
        check(f"{tag}: global indices identical", torch.equal(idx, ref_idx))
    check(f"{tag}: visual_token_length identical",
          mass_cfg.visual_token_length == ref_cfg.visual_token_length)
    m = mass_cfg.token_mass
    check(f"{tag}: one mass per delivered token", m is not None and m.numel() == tok.shape[0])
    check(f"{tag}: masses are positive integers", m is not None and bool((m >= 1).all()) and m.dtype == torch.long)
    off_cfg = published_qwen25(**cfg_kw)
    off_cfg.mass = False
    tok_off, _ = flashvid_compression_with_mass(feats.clone(), cls.clone(), off_cfg)
    check(f"{tag}: mass=False leaves token_mass unset, tokens unchanged",
          off_cfg.token_mass is None and torch.equal(tok_off, ref_tok))
    return mass_cfg


def main():
    # 1. Equality with FlashVID across seeds and the segmenting / non-segmenting paths.
    for seed in (0, 1, 2):
        feats, cls = video(seed)
        compare(f"seed {seed} segmented", feats, cls, {})
    feats, cls = video(7)
    compare("do_segment=False", feats, cls, {"do_segment": False})
    feats, cls = video(8, scene_every=1, drift=1.0)
    compare("every frame a new scene (single-frame segments)", feats, cls, {})
    feats, cls = video(9)
    compare("R=20%", feats, cls, {"retention_ratio": 0.20})

    # 2. Group sizes against a tree built by hand. Two frames, two tokens each,
    # no ADTS tokens: frame-1 token 0 is a copy of frame-0 token 1 and merges into
    # it; frame-1 token 1 is new and stays. Frame 0 keeps both of its tokens.
    f0 = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    f1 = torch.tensor([[0.0, 1.0], [1.0, -1.0]])
    feats = torch.stack([f0, f1])
    mask = torch.ones(2, 2, dtype=torch.bool)
    cfg = types.SimpleNamespace(num_attn_div_tokens=0, num_sttm_tokens=0)
    toks, idxs, sizes = _spatiotemporal_compression_with_mass(feats.clone(), 0.8, mask, cfg)
    check("hand tree: frame 0 keeps both tokens, frame 1 keeps one",
          [i.tolist() for i in idxs] == [[0, 1], [1]], str([i.tolist() for i in idxs]))
    check("hand tree: sizes [[1, 2], [1]]", [s.tolist() for s in sizes] == [[1, 2], [1]],
          str([s.tolist() for s in sizes]))

    # 3. Prefill bias, then pruning keeps each surviving key's own log m.
    n_before, n_vis, n_after, d = 3, 6, 4, 8
    L = n_before + n_vis + n_after
    cfg = published_qwen25(llm_retention_ratio=0.5)
    cfg.visual_token_start_index = n_before
    cfg.visual_token_length = n_vis
    cfg.token_mass = torch.tensor([1, 2, 3, 4, 5, 6])
    hidden = torch.zeros(1, L, d)
    mask = fvmass_score_bias(None, hidden, torch.arange(L), cfg)
    check("prefill: bias materialized on the visual keys",
          mask is not None and torch.allclose(mask[0, 0, -1, n_before:n_before + n_vis],
                                              torch.log(torch.arange(1.0, 7.0))))
    attn = torch.zeros(1, 2, L, L)
    for j in (1, 3, 4):   # visual positions 1, 3, 4 get the last query's attention
        attn[:, :, -1, n_before + j] = 1.0
    pos_emb = (torch.zeros(1, L, d), torch.zeros(1, L, d))
    _, pruned_mask, _, _, _, keep = fvmass_prune(
        hidden_states=hidden, causal_mask=mask, attentions=attn, cache_position=torch.arange(L),
        position_ids=torch.arange(L).unsqueeze(0), position_embeddings=pos_emb, flashvid_config=cfg)
    k = math.ceil(n_vis * 0.5)
    check("prune: records that inner pruning ran", getattr(cfg, "fvmass_pruned", None) is True)
    check("prune: kept masses are those of visual positions 1, 3, 4",
          cfg.token_mass.tolist() == [2, 4, 5], str(cfg.token_mass.tolist()))
    new_len = n_before + k + n_after
    check("prune: mask shrinks to the kept sequence", tuple(pruned_mask.shape) == (1, 1, new_len, new_len))
    last = pruned_mask[0, 0, -1]
    check("prune: last query sees log m of the KEPT tokens on the visual keys",
          torch.allclose(last[n_before:n_before + k], torch.log(torch.tensor([2.0, 4.0, 5.0]))),
          str(last[n_before:n_before + k].tolist()))
    check("prune: no bias on text keys",
          torch.equal(last[:n_before], torch.zeros(n_before)) and torch.equal(last[n_before + k:], torch.zeros(n_after)))
    floor = torch.finfo(pruned_mask.dtype).min
    upper = torch.ones(new_len, new_len, dtype=torch.bool).triu(1)
    check("prune: causal structure preserved", bool((pruned_mask[0, 0][upper] <= floor / 2).all())
          and bool((pruned_mask[0, 0][~upper] > floor / 2).all()))

    # 4. Decode after inner pruning is left alone (see the module docstring for why).
    dec_mask = torch.zeros(1, 1, 1, new_len + 1)
    check("decode after pruning: mask returned untouched",
          fvmass_score_bias(dec_mask, torch.zeros(1, 1, d), torch.tensor([99]), cfg) is dec_mask)
    check("decode after pruning: None stays None",
          fvmass_score_bias(None, torch.zeros(1, 1, d), torch.tensor([99]), cfg) is None)

    # 4b. Inner pruning off (pruning_layer past the last layer): every layer caches
    # the same L keys, so the decode query sees log m on the visual keys, as in prefill.
    def no_prune_cfg(mass):
        c = published_qwen25(pruning_layer=999)
        c.visual_token_start_index = n_before
        c.visual_token_length = n_vis
        c.token_mass = mass
        c.fvmass_pruned = False   # what flashvid_compression_with_mass leaves before prefill
        return c
    cfg_np = no_prune_cfg(torch.tensor([1, 2, 3, 4, 5, 6]))
    cache = types.SimpleNamespace(get_seq_length=lambda layer_idx=0: L)
    dec = fvmass_score_bias(None, torch.zeros(1, 1, d), torch.tensor([99]), cfg_np, cache)
    check("decode, no pruning: one query over the L cached keys plus its own",
          dec is not None and tuple(dec.shape) == (1, 1, 1, L + 1), None if dec is None else str(tuple(dec.shape)))
    if dec is not None and tuple(dec.shape) == (1, 1, 1, L + 1):
        row = dec[0, 0, 0]
        check("decode, no pruning: log m on the visual keys",
              torch.allclose(row[n_before:n_before + n_vis], torch.log(torch.arange(1.0, 7.0))), str(row.tolist()))
        check("decode, no pruning: text keys and the new token visible and unbiased",
              torch.equal(row[:n_before], torch.zeros(n_before))
              and torch.equal(row[n_before + n_vis:], torch.zeros(n_after + 1)))
        given = torch.zeros(1, 1, 1, L + 1)
        check("decode, no pruning: an explicit decode mask gets the same bias",
              torch.equal(fvmass_score_bias(given, torch.zeros(1, 1, d), torch.tensor([99]), cfg_np), dec))
    ctrl_np = no_prune_cfg(None)
    given = torch.zeros(1, 1, 1, L + 1)
    check("decode, no pruning, no mass (control arm): mask untouched",
          fvmass_score_bias(given, torch.zeros(1, 1, d), torch.tensor([99]), ctrl_np) is given)

    # 5. No mass (the control arm): pruning is exactly fastv_prune's.
    cfg_off = published_qwen25(llm_retention_ratio=0.5)
    cfg_off.visual_token_start_index = n_before
    cfg_off.visual_token_length = n_vis
    cfg_off.token_mass = None
    plain = apply_mass_bias(None, hidden, torch.arange(L), cfg_off)
    out = fvmass_prune(hidden_states=hidden, causal_mask=plain, attentions=attn, cache_position=torch.arange(L),
                       position_ids=torch.arange(L).unsqueeze(0), position_embeddings=pos_emb,
                       flashvid_config=cfg_off)
    check("control: no mass means the mask passes through fastv_prune untouched", out[1] is None)

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
