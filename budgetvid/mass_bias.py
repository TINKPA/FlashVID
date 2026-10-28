"""The mass channel: beta = log m as an additive attention-score bias.

Spec: notes/2026-08-28_method_budgetvid2_v1.html, eq (5) / L5 / B.6.

A quantized token stands for ``m`` originals. If it enters the softmax with
weight 1, the mode it represents is under-weighted by exactly that factor, and
no amount of good grouping fixes it. Adding ``log m`` to the score of every
visual key is not an approximation of the uncompressed attention -- it is an
algebraic identity with it:

    softmax_j(q.k_j + log m_j) V  =  sum_j m_j e^{q.k_j} v_j / sum_j m_j e^{q.k_j}

which is why this lands at every layer, head and query rather than being folded
into the features.

Implementation note (the one constraint the module imposes): this rides on the
additive float attention mask, so the model must run under an attention
implementation that accepts one -- ``sdpa`` or ``eager``, not
``flash_attention_2``. ``budgetvid()`` checks that up front rather than letting
it surface as a wrong number.
"""

from __future__ import annotations

import torch


def apply_mass_bias(causal_mask, hidden_states, cache_position, flashvid_config,
                    past_key_values=None):
    """Add ``log m`` to the scores of the compressed visual keys.

    Registered as method ``bv``'s score bias, so it is called once per decoder
    forward with the mask every layer is about to receive. Prefill and decode
    both come through here; the visual span is a contiguous slice of the
    sequence in both, because compression rewrote the sequence before the LLM
    ever saw it.

    Args:
        causal_mask: what ``create_causal_mask`` returned -- ``None`` (the
            implementation is using an implicit causal mask), a bool mask
            (True = attend), or an additive float mask.
        hidden_states: [bsz, q_len, d], for dtype/device/length.
        cache_position: [q_len] positions of the queries. NOTE these are the
            ORIGINAL sequence positions, not indices into the compressed
            sequence: the patched forward slices cache_position by the kept
            global indices without renumbering, so on a compressed video its
            last value is ~N, not ~B. It is therefore usable for counting
            queries and nothing else here.
        flashvid_config: carries ``token_mass``, ``visual_token_start_index``
            and ``visual_token_length``.
        past_key_values: only needed on the fallback path, to learn how many
            keys precede this forward's queries.

    Returns:
        The mask to hand the decoder layers: unchanged when there is no mass to
        apply, otherwise an additive float mask carrying the bias.
    """
    m = getattr(flashvid_config, "token_mass", None)
    if m is None:
        return causal_mask

    n_vis = int(m.numel())
    dtype, device = hidden_states.dtype, hidden_states.device
    bsz, q_len = hidden_states.shape[0], hidden_states.shape[1]
    if causal_mask is not None:
        kv_len = causal_mask.shape[-1]
    else:
        past = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = int(past) + q_len

    from .visual_positions import positions_of
    pos = positions_of(flashvid_config, n_vis, device=device)   # contiguous unless Qwen3 recorded them
    if n_vis and int(pos.max()) >= kv_len:
        raise ValueError(
            f"visual keys up to {int(pos.max())} do not fit a {kv_len}-token "
            "sequence; the mass vector and the compressed sequence disagree")

    if causal_mask is None:
        # The implementation was going to rely on an implicit causal mask, so
        # materialize it. Cheap here: the sequence is already compressed, so
        # this is a few hundred tokens squared, not the raw visual stream.
        # Over KEY SLOTS, not over cache_position's values: after compression
        # those values are original positions and index nothing in this
        # sequence. Query i of this forward sits at slot (kv_len - q_len + i).
        q_slot = torch.arange(kv_len - q_len, kv_len, device=device)
        allowed = q_slot[:, None] >= torch.arange(kv_len, device=device)[None, :]
        mask = torch.zeros(bsz, 1, q_len, kv_len, dtype=dtype, device=device)
        mask.masked_fill_(~allowed[None, None], torch.finfo(dtype).min)
    elif causal_mask.dtype == torch.bool:
        mask = torch.zeros(causal_mask.shape, dtype=dtype, device=device)
        mask.masked_fill_(~causal_mask, torch.finfo(dtype).min)
    else:
        mask = causal_mask.clone().to(dtype)

    beta = bias_mass(flashvid_config).to(device=device).to(dtype)
    mask[..., pos] += beta
    return mask


def bias_mass(flashvid_config) -> torch.Tensor:
    """The beta vector the bias applies: log of the (shaped) mass.

    The TRUE mass (``token_mass``) is what audits and dumps see; the bias may
    read a reshaped view of it:
      - ``token_mass_eff`` (set by the pipeline under ``mass_eff=True``): the
        purity-weighted effective mass, same length and order as ``token_mass``.
      - ``mass_cap > 0``: beta = log(min(m, cap)) -- the G2TM-motivated cap on
        how much weight a big group may claim.
    Returns a float32 tensor shaped like ``token_mass``.
    """
    m = flashvid_config.token_mass
    me = getattr(flashvid_config, "token_mass_eff", None)
    src = me if me is not None else m
    src = src.to(torch.float32).clamp(min=1.0)
    cap = float(getattr(flashvid_config, "mass_cap", 0.0) or 0.0)
    if cap > 0:
        src = src.clamp(max=cap)
    return src.log()


def clear_mass(flashvid_config) -> None:
    """Forget the previous video's masses.

    Called before every compression. Without it a run that switches policy
    mid-process, or a video whose compression is skipped, would inherit the
    previous sample's mass vector and bias the wrong keys.
    """
    flashvid_config.token_mass = None
    flashvid_config.token_mass_eff = None
    flashvid_config.visual_token_positions = None
