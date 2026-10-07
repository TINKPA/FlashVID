"""The inner-LLM pruning stage for method ``bv``, and the score bias that goes with it.

``llm_prune="none"`` (default) keeps everything: MassVID has one budget by
construction, B tokens at the LLM input held at every layer.

``llm_prune="fastv"`` runs FlashVID's second stage unchanged -- FastV
(``flashvid.utils.fastv_prune``): at layer ``pruning_layer`` the last query's
attention from layer ``pruning_layer - 1``, averaged over heads, ranks the
visual tokens and ``ceil(llm_retention_ratio * V)`` of them survive into the
remaining layers; text tokens are never touched. With FlashVID's own settings
(pruning_layer=20, llm_retention_ratio=0.3) MassVID at r then matches FlashVID at
R = r / 1.25 layer for layer: the same token count in layers 0-19 and the same
count in layers 20-27.

What this module adds on top of ``fastv_prune`` is the mass channel: the
``token_mass`` vector is re-indexed to the kept visual keys and the prefill bias
mask is rebuilt on them, so ``softmax(q.k + log m)`` stays the identity it is.
Decode is the one place the bias is then switched off: layers 0-19 cache the
full visual span and layers 20-27 the pruned one, and the single additive mask a
decode step hands every layer cannot carry both lengths (``apply_mass_bias``
would size it from layer 0's cache). On a multiple-choice benchmark the answer
token is produced by the prefill query, so this costs little, but it is a real
limitation and runs with it on record ``inner_pruned`` in their config.

Per-sample state lives on the config (``token_mass``, ``inner_pruned``,
``visual_token_length``); every compression entry resets it before the LLM sees
the new video, so nothing can leak from the previous one.
"""

from __future__ import annotations

import torch

from .mass_bias import apply_mass_bias

LLM_PRUNE_MODES = ("none", "fastv")


def reset_inner_state(flashvid_config) -> None:
    """Forget the previous video's inner-LLM record. Called by every compression entry."""
    flashvid_config.inner_pruned = False
    flashvid_config._bv_llm_keep = None
    flashvid_config._bv_kept_g = None


def no_llm_pruning(hidden_states, causal_mask, attentions, cache_position,
                   position_ids, position_embeddings, flashvid_config,
                   visual_pos_masks=None):
    """Keep everything: keep_indices is every position, nothing else is touched."""
    keep = torch.arange(hidden_states.shape[1], device=hidden_states.device)
    if cache_position is None:
        cache_position = keep
    if position_ids is None:
        position_ids = keep.unsqueeze(0)
    return hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep


def fastv_prune_keep_mass(hidden_states, causal_mask, attentions, cache_position,
                          position_ids, position_embeddings, flashvid_config,
                          visual_pos_masks=None):
    """FlashVID's ``fastv_prune``, with the mass vector and the bias kept on the kept keys."""
    # Imported here, not at module top: flashvid.utils pulls transformers in,
    # and the laptop tests of the compression path must not need it.
    from flashvid.utils import fastv_prune

    flashvid_config.inner_pruned = True
    start = int(flashvid_config.visual_token_start_index)
    n_before = int(flashvid_config.visual_token_length)
    out = fastv_prune(
        hidden_states=hidden_states,
        causal_mask=causal_mask,
        attentions=attentions,
        cache_position=cache_position,
        position_ids=position_ids,
        position_embeddings=position_embeddings,
        flashvid_config=flashvid_config,
        visual_pos_masks=visual_pos_masks,
    )
    hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices = out
    in_span = (keep_indices >= start) & (keep_indices < start + n_before)
    ranks = keep_indices[in_span] - start          # kept visual tokens, as ranks into the input span
    if ranks.numel() != int(flashvid_config.visual_token_length):
        raise AssertionError(
            f"fastv prune: kept {ranks.numel()} visual tokens but the config says "
            f"{flashvid_config.visual_token_length}")
    flashvid_config._bv_llm_keep = ranks.detach().cpu()

    m = getattr(flashvid_config, "token_mass", None)
    if m is None:
        return out
    m_kept = m.to(keep_indices.device)[ranks]
    flashvid_config.token_mass = m_kept

    if causal_mask is None or causal_mask.dtype == torch.bool:
        raise AssertionError("fastv prune: expected the additive float mask the prefill bias built")
    floor = torch.finfo(causal_mask.dtype).min
    rebuilt = torch.zeros_like(causal_mask)
    rebuilt.masked_fill_(causal_mask <= floor / 2, floor)
    beta = m_kept.to(torch.float32).clamp(min=1.0).log().to(causal_mask.dtype)
    rebuilt[..., start:start + m_kept.numel()] += beta
    return hidden_states, rebuilt, position_ids, cache_position, position_embeddings, keep_indices


def bv_llm_pruning(hidden_states, causal_mask, attentions, cache_position,
                   position_ids, position_embeddings, flashvid_config,
                   visual_pos_masks=None):
    """Dispatch on ``config.llm_prune``; registered as method ``bv``'s inner-LLM stage."""
    mode = str(getattr(flashvid_config, "llm_prune", "none") or "none")
    if mode == "none":
        fn = no_llm_pruning
    elif mode == "fastv":
        fn = fastv_prune_keep_mass
    else:
        raise KeyError(f"unknown llm_prune '{mode}'; known: {LLM_PRUNE_MODES}")
    if not getattr(flashvid_config, "_bv_prune_logged", False):
        flashvid_config._bv_prune_logged = True
        print(f"[BV] inner-LLM stage: {mode}"
              + (f" (layer {flashvid_config.pruning_layer}, keep "
                 f"{flashvid_config.llm_retention_ratio} of the visual tokens)" if mode == "fastv" else ""),
              flush=True)
    n_in = int(getattr(flashvid_config, "visual_token_length", 0))
    out = fn(hidden_states, causal_mask, attentions, cache_position, position_ids,
             position_embeddings, flashvid_config, visual_pos_masks=visual_pos_masks)
    dump_dir = str(getattr(flashvid_config, "dump_dir", "") or "")
    if mode == "fastv" and dump_dir:
        _record_inner_keep(dump_dir, flashvid_config, n_in)
    return out


def _record_inner_keep(dump_dir, cfg, n_in) -> None:
    """Sibling ``<tag>__llm.npz``: which vision-kept tokens survived FastV, as
    ``kept_llm_g`` in the same global indexing as the compression record's
    ``kept_g`` (mirrors recording.wrap_flashvid_keep for FlashVID)."""
    from .recording import dump_record, make_tag

    g = getattr(cfg, "_bv_kept_g", None)
    ranks = getattr(cfg, "_bv_llm_keep", None)
    if g is None or ranks is None:
        return
    cfg._bv_kept_g = None                        # first prefill only
    meta = {"method": "bv", "stage": "inner_llm", "llm_prune": "fastv",
            "pruning_layer": int(getattr(cfg, "pruning_layer", -1)),
            "llm_retention_ratio": float(getattr(cfg, "llm_retention_ratio", -1.0)),
            "n_visual_in": n_in, "n_visual_kept": int(ranks.numel())}
    if g.numel() == n_in:
        dump_record(dump_dir, make_tag(cfg) + "__llm", {}, {"kept_llm_g": g[ranks]}, meta, None, cfg=cfg)
    else:
        dump_record(dump_dir, make_tag(cfg) + "__llm", {}, {},
                    {**meta, "error": f"kept_g has {g.numel()} tokens, LLM saw {n_in}"}, None, cfg=cfg)


def mass_score_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values=None):
    """``apply_mass_bias`` in prefill, and in decode unless inner pruning ran (module docstring)."""
    if hidden_states.shape[1] > 1:
        return apply_mass_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values)
    pruned = bool(getattr(flashvid_config, "inner_pruned", False))
    if not getattr(flashvid_config, "_bv_decode_logged", False):
        # Once per process: which of the two decode paths this run is on.
        flashvid_config._bv_decode_logged = True
        on = not pruned and getattr(flashvid_config, "token_mass", None) is not None
        print(f"[BV] decode: inner pruning ran {pruned}, log-mass bias {'on' if on else 'off'}", flush=True)
    if pruned:
        return causal_mask
    return apply_mass_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values)
