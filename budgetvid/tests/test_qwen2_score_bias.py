"""The log-mass bias on the LLaVA-OV language-model path (plain ``Qwen2Model``).

    uv run --no-project --python 3.11 --with torch --with "transformers==4.57.3" --with numpy python budgetvid/tests/test_qwen2_score_bias.py

``flashvid/modeling_qwen2.py`` is the patched decoder LLaVA-OneVision runs on. Until
it called ``dispatch.score_bias`` right after building the causal mask (as
``modeling_qwen2_5_vl.py`` already did), method ``bv`` on LLaVA-OV ran with the mass
channel silently off: a plausible number, no error. This drives a tiny real
``Qwen2Model`` through the patched forward, under sdpa and under eager, and checks

1. no ``token_mass`` -> the patched model reproduces the UNPATCHED model to 1e-5,
   ``score_bias`` hands its input back untouched, and the mask every layer's
   attention receives is identical to what the unpatched model's attention got;
2. ``token_mass = m`` -> every layer's attention receives an additive float mask
   whose visual columns carry ``log m`` on the allowed positions, the text columns
   carry 0, the causal structure is intact, and the final hidden state differs;
3. the identity the channel rests on (``budgetvid/mass_bias.py``): attention with
   ``log m`` on key j equals unbiased attention over a sequence where key j is
   duplicated m_j times, checked on toy q/k/v through sdpa and through the softmax
   formula;
4. decode after a cached prefill: ``llm_prune="none"`` still biases the single
   decode query; ``llm_prune="fastv"`` (FastV really run at ``pruning_layer`` on
   the toy model) leaves the decode mask untouched, while the prefill mask after
   pruning carries the re-indexed masses.

``flashvid/__init__.py`` imports llava, which is not installed here, so the package
is registered as a bare namespace pointing at the source directory and its
``__init__`` never runs; the three ``bv`` registrations the forward needs are made
by hand, exactly as ``budgetvid/__init__.py`` makes them.
"""

import math
import pathlib
import sys
import types

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
if "flashvid" not in sys.modules:
    _pkg = types.ModuleType("flashvid")
    _pkg.__path__ = [str(ROOT / "flashvid")]
    sys.modules["flashvid"] = _pkg

from transformers.cache_utils import DynamicCache                                  # noqa: E402
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config              # noqa: E402
from transformers.models.qwen2.modeling_qwen2 import (                             # noqa: E402
    Qwen2Attention, Qwen2DecoderLayer, Qwen2Model)

import flashvid.modeling_qwen2 as mq                                               # noqa: E402
from flashvid.configuration_flashvid import FlashVidConfig                         # noqa: E402
from flashvid.dispatch import register_llm_pruning, register_score_bias            # noqa: E402
from budgetvid.llm_prune import bv_llm_pruning, mass_score_bias, reset_inner_state # noqa: E402
from budgetvid.mass_bias import apply_mass_bias                                    # noqa: E402

# The two registrations the patched forward dispatches on for method "bv"
# (budgetvid/__init__.py:_load_heavy makes the same two, plus compression).
register_llm_pruning("bv", bv_llm_pruning)
register_score_bias("bv", mass_score_bias)

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


# ---------------------------------------------------------------- fixtures

ORIG = {
    "attn": Qwen2Attention.forward,
    "layer": Qwen2DecoderLayer.forward,
    "model": Qwen2Model.forward,
}
N_BEFORE, V, N_AFTER = 2, 6, 3
T = N_BEFORE + V + N_AFTER
MASS = torch.tensor([1, 3, 1, 5, 2, 4])
assert MASS.numel() == V
LOG_M = MASS.float().clamp(min=1).log()
N_LAYERS, HIDDEN = 4, 64
PRUNING_LAYER, LLM_KEEP = 3, 0.5


def unpatch():
    Qwen2Attention.forward = ORIG["attn"]
    Qwen2DecoderLayer.forward = ORIG["layer"]
    Qwen2Model.forward = ORIG["model"]


def patch():
    # Exactly what flashvid/__init__.py does for LlavaQwenForCausalLM's text stack.
    Qwen2Attention.forward = mq.Qwen2Attention_forward
    Qwen2DecoderLayer.forward = mq.Qwen2DecoderLayer_forward
    Qwen2Model.forward = mq.Qwen2Model_forward


def build_model(impl):
    torch.manual_seed(0)
    cfg = Qwen2Config(hidden_size=HIDDEN, num_hidden_layers=N_LAYERS, num_attention_heads=4,
                      num_key_value_heads=2, intermediate_size=128, vocab_size=256,
                      max_position_embeddings=512, use_sliding_window=False)
    cfg._attn_implementation = impl
    model = Qwen2Model(cfg).eval()
    return model


def fv_config(token_mass, llm_prune="none"):
    c = FlashVidConfig(pruning_layer=PRUNING_LAYER, llm_retention_ratio=LLM_KEEP)
    c.method = "bv"
    c.llm_prune = llm_prune
    c.visual_token_start_index = N_BEFORE
    c.visual_token_length = V
    reset_inner_state(c)
    c.token_mass = None if token_mass is None else token_mass.clone()
    return c


class AttnMaskRecorder:
    """Wrap whatever ``Qwen2Attention.forward`` currently is; keep the mask of each call."""

    def __init__(self):
        self.masks = []

    def __enter__(self):
        real = Qwen2Attention.forward
        rec = self

        def wrapper(self_, hidden_states, position_embeddings, attention_mask, *a, **kw):
            rec.masks.append(None if attention_mask is None else attention_mask.detach().clone())
            return real(self_, hidden_states, position_embeddings, attention_mask, *a, **kw)

        self._real = real
        Qwen2Attention.forward = wrapper
        return self

    def __exit__(self, *exc):
        Qwen2Attention.forward = self._real


class ScoreBiasRecorder:
    """Wrap the ``score_bias`` name the patched forward imported; keep (in, out) per call."""

    def __init__(self):
        self.calls = []

    def __enter__(self):
        real = mq.score_bias
        rec = self

        def wrapper(causal_mask, *a, **kw):
            out = real(causal_mask, *a, **kw)
            rec.calls.append((causal_mask, out))
            return out

        self._real = real
        mq.score_bias = wrapper
        return self

    def __exit__(self, *exc):
        mq.score_bias = self._real


def same_mask(a, b):
    if a is None or b is None:
        return a is None and b is None
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def bias_layout_ok(mask, start, log_m, allowed, text_cols):
    """Float additive mask [1,1,q,kv]: log m on the visual columns where allowed, 0 on
    the text columns where allowed, <= finfo.min/2 where not allowed."""
    if mask is None or mask.dtype == torch.bool or mask.dim() != 4:
        return False, f"mask is {None if mask is None else (mask.dtype, tuple(mask.shape))}"
    m = mask[0, 0]
    floor = torch.finfo(m.dtype).min
    n_vis = log_m.numel()
    want = torch.zeros_like(m)
    want[:, start:start + n_vis] = log_m.to(m.dtype)
    if not torch.allclose(m[allowed], want[allowed], atol=1e-6):
        return False, f"allowed entries off by {float((m[allowed] - want[allowed]).abs().max()):.3g}"
    if not bool((m[~allowed] <= floor / 2).all()):
        return False, "a causally masked entry is not at the floor"
    return True, ""


def prefill_allowed(q_len, kv_len):
    q_slot = torch.arange(kv_len - q_len, kv_len)
    return q_slot[:, None] >= torch.arange(kv_len)[None, :]


# ---------------------------------------------------------------- the suites

def suite_model(impl):
    tag = f"[{impl}]"
    unpatch()
    model = build_model(impl)
    check(f"{tag} model built under {impl}", model.config._attn_implementation == impl,
          str(model.config._attn_implementation))
    torch.manual_seed(1)
    x = torch.randn(1, T, HIDDEN)
    x_dec = torch.randn(1, 1, HIDDEN)

    # Reference: the UNPATCHED Qwen2Model, with the masks its attention receives.
    with torch.no_grad(), AttnMaskRecorder() as ref_rec:
        ref = model(inputs_embeds=x, use_cache=False).last_hidden_state
    ref_masks = ref_rec.masks
    check(f"{tag} reference run touched every layer's attention once", len(ref_masks) == N_LAYERS)

    patch()

    # 1. No mass: patched == unpatched, and the bias stage is the identity.
    model.flashvid_config = fv_config(None)
    with torch.no_grad(), AttnMaskRecorder() as rec, ScoreBiasRecorder() as sb:
        out0 = model(inputs_embeds=x, use_cache=False).last_hidden_state
    check(f"{tag} no mass: patched forward == unpatched forward (1e-5)",
          torch.allclose(out0, ref, atol=1e-5, rtol=0), f"max |diff| {float((out0 - ref).abs().max()):.3g}")
    check(f"{tag} no mass: score_bias called once per forward", len(sb.calls) == 1, str(len(sb.calls)))
    check(f"{tag} no mass: score_bias returns its input object untouched",
          len(sb.calls) == 1 and sb.calls[0][1] is sb.calls[0][0])
    check(f"{tag} no mass: every layer's attention gets the same mask as unpatched",
          len(rec.masks) == N_LAYERS and all(same_mask(a, b) for a, b in zip(rec.masks, ref_masks)),
          f"ref {[None if m is None else tuple(m.shape) for m in ref_masks]} vs "
          f"patched {[None if m is None else tuple(m.shape) for m in rec.masks]}")
    check(f"{tag} no mass: prune stage left the config untouched",
          model.flashvid_config.inner_pruned is False and model.flashvid_config.visual_token_length == V)

    # 2. Mass: every layer sees log m on the visual keys; the output moves.
    model.flashvid_config = fv_config(MASS)
    with torch.no_grad(), AttnMaskRecorder() as rec, ScoreBiasRecorder() as sb:
        out1 = model(inputs_embeds=x, use_cache=False).last_hidden_state
    check(f"{tag} mass: score_bias materialized a float mask",
          len(sb.calls) == 1 and sb.calls[0][1] is not None and sb.calls[0][1].dtype == x.dtype)
    allowed = prefill_allowed(T, T)
    text_cols = list(range(N_BEFORE)) + list(range(N_BEFORE + V, T))
    for li, m in enumerate(rec.masks):
        ok, why = bias_layout_ok(m, N_BEFORE, LOG_M, allowed, text_cols)
        check(f"{tag} mass: layer {li} attention mask = causal + log m on keys [{N_BEFORE},{N_BEFORE + V})", ok, why)
    check(f"{tag} mass: final hidden state differs from the no-mass run",
          float((out1 - out0).abs().max()) > 1e-4, f"max |diff| {float((out1 - out0).abs().max()):.3g}")
    check(f"{tag} mass: text-only prefix (before any visual key) unchanged",
          torch.allclose(out1[:, :N_BEFORE], out0[:, :N_BEFORE], atol=1e-5, rtol=0),
          f"max |diff| {float((out1[:, :N_BEFORE] - out0[:, :N_BEFORE]).abs().max()):.3g}")

    # 4a. Decode with llm_prune="none": the single decode query is still biased.
    model.flashvid_config = fv_config(MASS)
    with torch.no_grad():
        pre = model(inputs_embeds=x, use_cache=True)
    cache = pre.past_key_values
    check(f"{tag} decode/none: prefill returned a cache of {T} keys",
          isinstance(cache, DynamicCache) and cache.get_seq_length() == T)
    with torch.no_grad(), AttnMaskRecorder() as rec, ScoreBiasRecorder() as sb:
        model(inputs_embeds=x_dec, past_key_values=cache, use_cache=True)
    check(f"{tag} decode/none: score_bias produced a float mask for the decode query",
          len(sb.calls) == 1 and sb.calls[0][1] is not None and tuple(sb.calls[0][1].shape) == (1, 1, 1, T + 1),
          None if not sb.calls or sb.calls[0][1] is None else str(tuple(sb.calls[0][1].shape)))
    dec_allowed = torch.ones(1, T + 1, dtype=torch.bool)
    for li, m in enumerate(rec.masks):
        ok, why = bias_layout_ok(m, N_BEFORE, LOG_M, dec_allowed, text_cols + [T])
        check(f"{tag} decode/none: layer {li} decode mask carries log m over the cached visual keys", ok, why)

    # 4b. Decode with llm_prune="fastv": FastV runs at pruning_layer in prefill, the
    # decode mask is then passed through untouched.
    model.flashvid_config = cfg = fv_config(MASS, llm_prune="fastv")
    k = math.ceil(V * LLM_KEEP)
    T2 = T - (V - k)
    fastv_ran = False
    try:
        with torch.no_grad(), AttnMaskRecorder() as rec:
            pre = model(inputs_embeds=x, use_cache=True)
        fastv_ran = True
    except Exception as e:  # report exactly what happened; the decode check below falls back
        import traceback
        print(f"NOTE  {tag} fastv prefill raised on the toy model: {type(e).__name__}: {e}")
        traceback.print_exc()
    if fastv_ran:
        cache = pre.past_key_values
        check(f"{tag} decode/fastv: prefill recorded inner pruning", cfg.inner_pruned is True)
        check(f"{tag} decode/fastv: prefill kept ceil({LLM_KEEP} * {V}) = {k} visual tokens",
              cfg.visual_token_length == k and cfg.token_mass is not None and cfg.token_mass.numel() == k,
              f"len {cfg.visual_token_length}, mass {None if cfg.token_mass is None else cfg.token_mass.tolist()}")
        kept = getattr(cfg, "_bv_llm_keep", None)
        check(f"{tag} decode/fastv: kept masses are MASS[kept ranks]",
              kept is not None and cfg.token_mass.tolist() == MASS[kept].tolist())
        shapes = [None if m is None else tuple(m.shape) for m in rec.masks]
        check(f"{tag} decode/fastv: layers 0-{PRUNING_LAYER - 1} saw {T} keys, layer {PRUNING_LAYER} saw {T2}",
              len(rec.masks) == N_LAYERS and all(s == (1, 1, T, T) for s in shapes[:PRUNING_LAYER])
              and all(s == (1, 1, T2, T2) for s in shapes[PRUNING_LAYER:]), str(shapes))
        if len(rec.masks) == N_LAYERS and rec.masks[PRUNING_LAYER] is not None:
            log_kept = cfg.token_mass.float().clamp(min=1).log()
            ok, why = bias_layout_ok(rec.masks[PRUNING_LAYER], N_BEFORE, log_kept, prefill_allowed(T2, T2),
                                     list(range(N_BEFORE)) + list(range(N_BEFORE + k, T2)))
            check(f"{tag} decode/fastv: layer {PRUNING_LAYER} prefill mask carries the re-indexed log m", ok, why)
        check(f"{tag} decode/fastv: cache holds {T} keys at layer 0 and {T2} at layer {PRUNING_LAYER}",
              cache.get_seq_length(0) == T and cache.get_seq_length(PRUNING_LAYER) == T2,
              f"{cache.get_seq_length(0)}, {cache.get_seq_length(PRUNING_LAYER)}")
    else:
        # Fallback: a cached prefill without pruning, then the pruned flag set by hand.
        model.flashvid_config = cfg = fv_config(MASS)
        with torch.no_grad():
            pre = model(inputs_embeds=x, use_cache=True)
        cache = pre.past_key_values
        cfg.inner_pruned = True
        print(f"NOTE  {tag} decode/fastv check runs with inner_pruned=True set by hand")
    with torch.no_grad(), AttnMaskRecorder() as rec, ScoreBiasRecorder() as sb:
        model(inputs_embeds=x_dec, past_key_values=cache, use_cache=True)
    check(f"{tag} decode/fastv: score_bias hands the decode mask back untouched",
          len(sb.calls) == 1 and sb.calls[0][1] is sb.calls[0][0])
    no_bias = all(m is None or bool((m[..., N_BEFORE:N_BEFORE + k] == 0).all()) for m in rec.masks)
    check(f"{tag} decode/fastv: no layer's decode mask carries a bias", len(rec.masks) == N_LAYERS and no_bias)

    unpatch()


def suite_identity():
    """(3) log m on key j == duplicating key j m_j times; toy q/k/v, d=8, one head."""
    torch.manual_seed(2)
    d = 8
    q = torch.randn(1, 1, T, d)
    k = torch.randn(1, 1, T, d)
    v = torch.randn(1, 1, T, d)
    cfg = types.SimpleNamespace(token_mass=MASS.clone(), visual_token_start_index=N_BEFORE,
                                visual_token_length=V)
    mask = apply_mass_bias(None, torch.zeros(1, T, d), torch.arange(T), cfg)
    check("identity: apply_mass_bias on None materializes [1,1,T,T] float", mask is not None
          and tuple(mask.shape) == (1, 1, T, T) and mask.dtype == torch.float32)
    biased = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)[0, 0, -1]
    idx = torch.cat([torch.arange(N_BEFORE),
                     torch.repeat_interleave(torch.arange(N_BEFORE, N_BEFORE + V), MASS),
                     torch.arange(N_BEFORE + V, T)])
    dup = F.scaled_dot_product_attention(q[..., -1:, :], k[..., idx, :], v[..., idx, :])[0, 0, 0]
    check("identity (sdpa): biased attention == attention over the duplicated keys",
          torch.allclose(biased, dup, atol=1e-5), f"max |diff| {float((biased - dup).abs().max()):.3g}")
    s = (q[0, 0, -1] @ k[0, 0].T) / math.sqrt(d)
    w = torch.ones(T)
    w[N_BEFORE:N_BEFORE + V] = MASS.float()
    closed = (w * s.exp()) @ v[0, 0] / (w * s.exp()).sum()
    check("identity (formula): softmax(s + log m) v == sum m e^s v / sum m e^s",
          torch.allclose(biased, closed, atol=1e-5), f"max |diff| {float((biased - closed).abs().max()):.3g}")
    # Earlier queries: the bias changes them too (every query, not only the last).
    plain = F.scaled_dot_product_attention(q, k, v, is_causal=True)[0, 0]
    full = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)[0, 0]
    check("identity: queries before the visual span are unchanged, later ones move",
          torch.allclose(plain[:N_BEFORE], full[:N_BEFORE], atol=1e-6)
          and float((plain[N_BEFORE + 1:] - full[N_BEFORE + 1:]).abs().max()) > 1e-4)
    # A mass of 1 everywhere is a no-op on the scores.
    cfg1 = types.SimpleNamespace(token_mass=torch.ones(V, dtype=torch.long), visual_token_start_index=N_BEFORE,
                                 visual_token_length=V)
    m1 = apply_mass_bias(None, torch.zeros(1, T, d), torch.arange(T), cfg1)
    check("identity: all-ones mass == plain causal attention",
          torch.allclose(F.scaled_dot_product_attention(q, k, v, attn_mask=m1)[0, 0], plain, atol=1e-6))


def main():
    print(f"torch {torch.__version__}, sequence: {N_BEFORE} text + {V} visual + {N_AFTER} text = {T}, "
          f"mass {MASS.tolist()}, pruning_layer {PRUNING_LAYER} of {N_LAYERS}")
    suite_identity()
    for impl in ("sdpa", "eager"):
        suite_model(impl)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
