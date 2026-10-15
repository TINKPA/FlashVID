"""FastV at any ``pruning_layer`` K on the Qwen2.5-VL text stack, method ``bv`` with the mass channel.

    uv run --no-project --python 3.11 --with torch --with "transformers==4.57.3" --with numpy python budgetvid/tests/test_pruning_layer.py

Every run before 2026-10-15 used K=20, and ``test_llm_prune.py`` / ``test_qwen2_score_bias.py``
only ever exercised that one layer (or a 4-layer toy). The (K, k) sweep
(``notes/2026-10-15_experiment_vmme-fastv-sweep.html``) moves K to 2 / 8 / 14 / 24, and K enters
the code only through the patched decoder loop (``flashvid/modeling_qwen2_5_vl.py``): layer K-1
is asked for attention weights, layer K prunes. ``bv_llm_pruning`` itself never sees K.

This drives a tiny real 28-layer ``Qwen2_5_VLTextModel`` through that patched forward, under sdpa
(what the mq rows run) and eager, for K in {1, 2, 8, 14, 20, 24, 27} x k in {0, 0.3, 0.5}, and checks

1. layers 0..K-1 see T keys and layers K..27 see T - V + ceil(k*V) (the attention masks each
   layer receives, and the prefill KV cache per layer);
2. the attention weights come from layer K-1 and only from it, and the kept visual tokens are
   exactly the top ceil(k*V) of that layer's last-query attention averaged over heads;
3. ``token_mass`` is re-indexed to the kept ranks and layer K's mask carries their log m;
4. the 28-layer visual-token average equals (K*V + (28-K)*ceil(k*V)) / 28 -- the budget
   identity r = 28R / (K + (28-K)k) the sweep is built on;

plus K=28 (past the last layer): no pruning, every layer sees T (the upstream loop still asks
layer 27 for weights it never uses).
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

from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLTextConfig   # noqa: E402
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (                           # noqa: E402
    Qwen2_5_VLAttention, Qwen2_5_VLTextModel)

import flashvid.modeling_qwen2_5_vl as mq25                                                 # noqa: E402
from flashvid.configuration_flashvid import FlashVidConfig                                  # noqa: E402
from flashvid.dispatch import register_llm_pruning, register_score_bias                     # noqa: E402
from budgetvid.llm_prune import bv_llm_pruning, mass_score_bias, reset_inner_state          # noqa: E402

register_llm_pruning("bv", bv_llm_pruning)
register_score_bias("bv", mass_score_bias)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    if not cond:
        print(f"FAIL  {name}" + (f"   [{detail}]" if detail else ""))


N_LAYERS, HIDDEN, HEADS, KV_HEADS = 28, 64, 4, 2
N_BEFORE, V, N_AFTER = 3, 40, 5
T = N_BEFORE + V + N_AFTER
KS = (1, 2, 8, 14, 20, 24, 27)
kS = (0.0, 0.3, 0.5)

torch.manual_seed(0)
MASS = torch.randint(1, 6, (V,))
TEXT_COLS = list(range(N_BEFORE)) + list(range(N_BEFORE + V, T))


def build_model(impl):
    torch.manual_seed(0)
    cfg = Qwen2_5_VLTextConfig(hidden_size=HIDDEN, num_hidden_layers=N_LAYERS, num_attention_heads=HEADS,
                               num_key_value_heads=KV_HEADS, intermediate_size=128, vocab_size=256,
                               max_position_embeddings=512, use_sliding_window=False,
                               rope_scaling={"type": "mrope", "mrope_section": [2, 3, 3]})
    cfg._attn_implementation = impl
    Qwen2_5_VLAttention.forward = mq25.Qwen2_5_VLAttention_forward
    Qwen2_5_VLTextModel.forward = mq25.Qwen2_5_VLTextModel_forward
    return Qwen2_5_VLTextModel(cfg).eval()


def fv_config(K, k):
    c = FlashVidConfig(pruning_layer=K, llm_retention_ratio=k)
    c.method = "bv"
    c.llm_prune = "fastv"
    c.visual_token_start_index = N_BEFORE
    c.visual_token_length = V
    reset_inner_state(c)
    c.token_mass = MASS.clone()
    return c


class Recorder:
    """Wrap the patched attention: per call keep (layer_idx, mask shape, mask, attn weights if returned)."""

    def __enter__(self):
        real = Qwen2_5_VLAttention.forward
        self.calls, self._real = [], real

        def wrapper(self_, hidden_states, attention_mask=None, *a, **kw):
            out = real(self_, hidden_states, attention_mask, *a, **kw)
            m = None if attention_mask is None else attention_mask.detach().clone()
            w = out[1].detach().clone() if kw.get("output_attentions") and out[1] is not None else None
            self.calls.append((self_.layer_idx, m, w))
            return out

        Qwen2_5_VLAttention.forward = wrapper
        return self

    def __exit__(self, *exc):
        Qwen2_5_VLAttention.forward = self._real


def run_case(model, K, k, x, pos):
    tag = f"[{model.config._attn_implementation} K={K} k={k:g}]"
    model.flashvid_config = cfg = fv_config(K, k)
    kept_n = math.ceil(k * V)
    T2 = T - V + kept_n
    with torch.no_grad(), Recorder() as rec:
        out = model(inputs_embeds=x, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    prunes = K < N_LAYERS

    # 1. per-layer sequence length: masks and the KV cache
    want = [T if (li < K or not prunes) else T2 for li in range(N_LAYERS)]
    got = [None if m is None else m.shape[-1] for _, m, _ in rec.calls]
    check(f"{tag} one attention call per layer", [li for li, _, _ in rec.calls] == list(range(N_LAYERS)),
          str([li for li, _, _ in rec.calls]))
    check(f"{tag} mask key length per layer", got == want, f"got {got}")
    cl = [cache.get_seq_length(li) for li in range(N_LAYERS)]
    check(f"{tag} KV cache length per layer", cl == want, f"got {cl}")
    check(f"{tag} inner_pruned recorded", cfg.inner_pruned is prunes)

    # 4. the 28-layer visual-token average (the sweep's budget identity)
    vis = [w - (T - V) for w in want]
    avg = sum(vis) / N_LAYERS
    check(f"{tag} layer-average visual tokens = (K*V + (28-K)*ceil(kV)) / 28",
          (not prunes) or abs(avg - (K * V + (N_LAYERS - K) * kept_n) / N_LAYERS) < 1e-12, f"{avg}")
    if not prunes:
        # Upstream loop still asks layer K-1 = 27 for weights; nothing consumes them.
        check(f"{tag} no pruning, mass unchanged", cfg.visual_token_length == V and torch.equal(cfg.token_mass, MASS))
        return avg

    # 2. weights from layer K-1 only; kept = top ceil(kV) of its last-query attention, head-averaged
    with_w = [li for li, _, w in rec.calls if w is not None]
    check(f"{tag} attention weights requested from layer K-1 only", with_w == [K - 1], str(with_w))
    w = dict((li, w) for li, _, w in rec.calls)[K - 1]
    score = w[0, :, -1, N_BEFORE:N_BEFORE + V].float().mean(0)
    expect = sorted(torch.topk(score, kept_n).indices.tolist()) if kept_n else []
    kept = [] if cfg._bv_llm_keep is None else sorted(cfg._bv_llm_keep.tolist())
    check(f"{tag} kept = top-{kept_n} of layer {K - 1}'s last-query attention", kept == expect,
          f"kept {kept} expect {expect}")
    check(f"{tag} config length = ceil(kV) = {kept_n}", cfg.visual_token_length == kept_n)

    # 3. mass re-indexed; layer K's mask carries the kept log m, text columns 0
    check(f"{tag} token_mass re-indexed to the kept ranks",
          cfg.token_mass.tolist() == MASS[torch.tensor(kept, dtype=torch.long)].tolist())
    mK = dict((li, m) for li, m, _ in rec.calls)[K]
    last = mK[0, 0, -1].float()
    ok_vis = torch.allclose(last[N_BEFORE:N_BEFORE + kept_n], cfg.token_mass.float().clamp(min=1).log(), atol=1e-5)
    ok_txt = bool((last[:N_BEFORE] == 0).all()) and bool((last[N_BEFORE + kept_n:] == 0).all())
    check(f"{tag} layer {K} mask: log m on kept visual keys, 0 on text", ok_vis and ok_txt)
    m0 = dict((li, m) for li, m, _ in rec.calls)[0][0, 0, -1].float()
    check(f"{tag} layer 0 mask: log m on all V visual keys",
          torch.allclose(m0[N_BEFORE:N_BEFORE + V], MASS.float().log(), atol=1e-5))
    return avg


def main():
    print(f"torch {torch.__version__}; {N_LAYERS} layers, sequence {N_BEFORE} text + {V} visual + {N_AFTER} text = {T}")
    torch.manual_seed(1)
    x = torch.randn(1, T, HIDDEN)
    pos = torch.arange(T).view(1, 1, T).expand(4, 1, T).contiguous()
    for impl in ("sdpa", "eager"):
        model = build_model(impl)
        rows = []
        for K in KS:
            for k in kS:
                rows.append((K, k, run_case(model, K, k, x, pos)))
        run_case(model, N_LAYERS, 0.3, x, pos)
        print(f"[{impl}] layer-average visual tokens (V={V}):")
        for K in KS:
            print("   K=%2d  " % K + "  ".join(f"k={k:g}: {a:6.2f}" for KK, k, a in rows if KK == K))
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
