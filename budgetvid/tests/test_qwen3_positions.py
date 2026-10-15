"""Qwen3-VL: the log-mass bias and FastV bookkeeping follow the TRUE visual positions.

    uv run --no-project --python 3.11 --with torch --with "transformers==4.57.3" --with numpy python budgetvid/tests/test_qwen3_positions.py

On Qwen3-VL the patched forward keeps every non-visual token of the video prompt, so after
compression non-visual tokens sit between the visual ones (2026-10-15 smoke: 219 positions
survived FastV inside [start, start+len), 144 of them visual). Before the fix the bias was
added on that contiguous span and the FastV assertion counted span positions.

This drives a tiny real Qwen3VLTextModel (sdpa and eager, 8 layers) through the patched
forward, on a sequence whose visual tokens are interleaved with non-visual ones, and checks

1. the expression the Qwen3 forward uses to record positions (searchsorted of the kept
   visual global indices into keep_global_indexes) gives the visual tokens' slots;
2. the prefill mask every layer before K receives carries log m on exactly the visual
   positions and 0 on the non-visual tokens between them;
3. FastV at layer K keeps the top ceil(k*V) visual tokens of layer K-1's last-query
   attention (by visual_pos_masks), the AssertionError of the smoke does not fire, the mass
   vector is re-indexed, visual_token_positions is remapped to the new slots, and layer K's
   mask carries the kept log m there and 0 elsewhere;
4. without visual_token_positions (contiguous backbones) positions_of is the old span.
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

from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig      # noqa: E402
from transformers.models.qwen3_vl.modeling_qwen3_vl import (                            # noqa: E402
    Qwen3VLTextAttention, Qwen3VLTextDecoderLayer, Qwen3VLTextModel)

import flashvid.modeling_qwen3_vl as mq3                                                 # noqa: E402
from flashvid.configuration_flashvid import FlashVidConfig                               # noqa: E402
from flashvid.dispatch import register_llm_pruning, register_score_bias                  # noqa: E402
from budgetvid.llm_prune import bv_llm_pruning, mass_score_bias, reset_inner_state       # noqa: E402
from budgetvid.visual_positions import positions_of                                      # noqa: E402

register_llm_pruning("bv", bv_llm_pruning)
register_score_bias("bv", mass_score_bias)
PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"FAIL  {name}" + (f"   [{detail}]" if detail else ""))


# 3 text | 5 x (4 visual + 2 non-visual) | 4 text
N_LAYERS, HIDDEN = 8, 64
LAYOUT = ["t"] * 3 + (["v"] * 4 + ["t"] * 2) * 5 + ["t"] * 4
T = len(LAYOUT)
VPOS = torch.tensor([i for i, c in enumerate(LAYOUT) if c == "v"])
V = VPOS.numel()
torch.manual_seed(0)
MASS = torch.randint(1, 7, (V,))
LOG_M = MASS.float().log()


def build(impl):
    torch.manual_seed(0)
    cfg = Qwen3VLTextConfig(hidden_size=HIDDEN, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                            num_key_value_heads=2, head_dim=16, intermediate_size=128, vocab_size=256,
                            max_position_embeddings=512,
                            rope_scaling={"rope_type": "default", "mrope_section": [2, 3, 3], "mrope_interleaved": True})
    cfg._attn_implementation = impl
    Qwen3VLTextAttention.forward = mq3.Qwen3VLTextAttention_forward
    Qwen3VLTextDecoderLayer.forward = mq3.Qwen3VLTextDecoderLayer_forward
    Qwen3VLTextModel.forward = mq3.Qwen3VLTextModel_forward
    return Qwen3VLTextModel(cfg).eval()


def fv_cfg(K, k):
    c = FlashVidConfig(pruning_layer=K, llm_retention_ratio=k)
    c.method, c.llm_prune = "bv", "fastv"
    c.visual_token_start_index, c.visual_token_length = int(VPOS[0]), V
    reset_inner_state(c)
    c.token_mass = MASS.clone()
    c.visual_token_positions = VPOS.clone()
    return c


class Rec:
    def __enter__(self):
        real = Qwen3VLTextAttention.forward
        self.calls, self._real = [], real

        def wrap(self_, hidden_states, position_embeddings, attention_mask, *a, **kw):
            out = real(self_, hidden_states, position_embeddings, attention_mask, *a, **kw)
            w = out[1].detach().clone() if kw.get("output_attentions") and out[1] is not None else None
            self.calls.append((self_.layer_idx, None if attention_mask is None else attention_mask.detach().clone(), w))
            return out

        Qwen3VLTextAttention.forward = wrap
        return self

    def __exit__(self, *e):
        Qwen3VLTextAttention.forward = self._real


def suite_record_expression():
    # original prompt: 2 text | 3 start/end-wrapped groups of 6 video tokens | 3 text; markers dropped
    ids = ["t", "t"] + (["s"] + ["v"] * 6 + ["e"]) * 3 + ["t"] * 3
    vis_idx = torch.tensor([i for i, c in enumerate(ids) if c == "v"])
    non_vis = torch.tensor([i for i, c in enumerate(ids) if c == "t"])
    keep_vis = torch.tensor([0, 2, 5, 6, 11, 17])            # sorted, as pipeline.py returns them
    keep_global = torch.cat([vis_idx[keep_vis], non_vis]).sort().values
    pos = torch.searchsorted(keep_global, vis_idx[keep_vis])  # the line in modeling_qwen3_vl.py
    check("record: positions index the kept visual tokens in the new sequence",
          torch.equal(keep_global[pos], vis_idx[keep_vis]), str(pos.tolist()))


def suite(impl):
    model = build(impl)
    x = torch.randn(1, T, HIDDEN)
    vmask = torch.zeros(1, T, dtype=torch.bool); vmask[0, VPOS] = True
    for K, k in ((3, 0.3), (5, 0.5), (2, 0.0)):
        tag = f"[{impl} K={K} k={k:g}]"
        model.flashvid_config = cfg = fv_cfg(K, k)
        try:
            with torch.no_grad(), Rec() as rec:
                model(inputs_embeds=x, use_cache=True, visual_pos_masks=vmask)
        except AssertionError as e:
            check(f"{tag} prefill runs (no span-count assertion)", False, str(e)); continue
        masks = {li: m for li, m, _ in rec.calls}
        last0 = masks[0][0, 0, -1].float()
        on_v = torch.allclose(last0[VPOS], LOG_M, atol=1e-5)
        nonv = torch.tensor([i for i in range(T) if i not in set(VPOS.tolist())])
        check(f"{tag} layer 0: log m on exactly the visual positions", on_v)
        check(f"{tag} layer 0: 0 on every non-visual key, incl. those between visual tokens",
              bool((last0[nonv] == 0).all()))
        n_keep = math.ceil(k * V)
        w = {li: w for li, _, w in rec.calls if w is not None}
        check(f"{tag} weights from layer K-1 only", list(w) == [K - 1], str(list(w)))
        score = w[K - 1][0, :, -1, :].float().mean(0)[VPOS]
        expect = sorted(torch.topk(score, n_keep).indices.tolist()) if n_keep else []
        got = sorted(cfg._bv_llm_keep.tolist()) if cfg._bv_llm_keep is not None else []
        check(f"{tag} kept = top-{n_keep} visual tokens of layer {K - 1}", got == expect, f"{got} vs {expect}")
        check(f"{tag} config length = ceil(kV)", cfg.visual_token_length == n_keep)
        check(f"{tag} mass re-indexed", cfg.token_mass.tolist() == MASS[torch.tensor(got, dtype=torch.long)].tolist())
        kept_old = VPOS[torch.tensor(got, dtype=torch.long)]
        keep_all = torch.cat([nonv, kept_old]).sort().values
        new_pos = torch.searchsorted(keep_all, kept_old)
        check(f"{tag} visual_token_positions remapped to the new slots",
              torch.equal(cfg.visual_token_positions.cpu(), new_pos), f"{cfg.visual_token_positions.tolist()} vs {new_pos.tolist()}")
        lastK = masks[K][0, 0, -1].float()
        T2 = keep_all.numel()
        check(f"{tag} layer {K} mask is {T2} keys wide", lastK.numel() == T2, str(lastK.numel()))
        ok_v = torch.allclose(lastK[new_pos], cfg.token_mass.float().clamp(min=1).log(), atol=1e-5)
        others = torch.tensor([i for i in range(T2) if i not in set(new_pos.tolist())])
        check(f"{tag} layer {K}: kept log m at the remapped positions, 0 elsewhere",
              ok_v and bool((lastK[others] == 0).all()))


def suite_fallback():
    c = FlashVidConfig(); c.visual_token_start_index, c.visual_token_length = 5, 7
    check("fallback: no positions recorded -> contiguous span",
          positions_of(c).tolist() == list(range(5, 12)))


def main():
    suite_record_expression()
    suite_fallback()
    for impl in ("sdpa", "eager"):
        suite(impl)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
