"""FlashVID with the mass channel: method ``fvmass``, ``budgetvid(policy="flashvid_mass")``.

A plug-in test of whether carrying group sizes into attention helps a published
merge-based compressor. Every decision stays FlashVID's. The three vision-side
functions below are copies of ``flashvid/utils.py``'s ``flashvid_compression``,
``segment_compression`` and ``spatiotemporal_compression`` with one addition:
each delivered token's group size (how many original tokens it averages) is
counted next to the tokens. The copies must return exactly what the originals
return -- ``budgetvid/tests/test_flashvid_mass.py`` checks token-for-token
equality -- so the ``mass=False`` arm IS FlashVID, on the same attention kernel.

Group sizes. An ADTS-selected token stands for itself (m = 1). A token TSTM
retains stands for its whole temporal tree. A DPC-kNN cluster stands for the sum
of its members' trees. FlashVID's own averaging uses a per-hop count that it
resets after every merge, so the tree sizes are tracked separately and never
reset. Whenever TSTM runs, FlashVID drops nothing on the vision side, so the
masses add up to the number of input tokens -- plus, in a single-frame segment,
the ADTS tokens that upstream also feeds to DPC-kNN and so delivers twice. The
sum each segment's path implies is asserted on every video.

Inner-LLM pruning is FlashVID's ``fastv_prune``, with two additions around it:
the mass vector is cut to the kept tokens, and the pruned mask is rebuilt so each
kept visual key carries its own ``log m``. ``fastv_prune`` slices the mask by its
first rows and columns, which is right for a pure causal mask and would put a
bias on the wrong keys.

When inner pruning runs, the bias is applied in PREFILL ONLY. After
``pruning_layer`` the later layers' KV caches hold fewer visual tokens than the
earlier layers', and decoding hands one mask to every layer, so no single mask
fits all layers during decode. The first generated token -- the answer letter on
the multiple-choice benchmarks -- comes out of the fully biased prefill; the
tokens after it are decoded without the bias.

When ``pruning_layer`` lies past the last layer, no pruning runs, every layer
caches the same keys, and decode carries the bias too, as method ``bv`` does.
Decode follows what actually happened to the sample: ``fvmass_prune`` records
that it ran, and every compression clears the record before the next prefill.
"""

from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from flashvid.utils import ALL_TOKEN_SELECTION_METHOD, dpc_knn, fastv_prune, segment

from .mass_bias import apply_mass_bias, clear_mass


def flashvid_compression_with_mass(video_features, cls_attention, flashvid_config):
    """``flashvid.utils.flashvid_compression`` plus ``flashvid_config.token_mass``.

    Returns exactly what the original returns. Sets ``token_mass`` (aligned with
    the returned, index-sorted tokens) when ``flashvid_config.mass`` is on and
    leaves it None otherwise.
    """
    clear_mass(flashvid_config)
    flashvid_config.fvmass_pruned = False
    num_frames, num_visual_tokens, feat_dim = video_features.shape

    if flashvid_config.do_segment:
        segment_lengths = segment(
            video_features=video_features.mean(1),
            segment_threshold=flashvid_config.segment_threshold,
            min_segment_num=flashvid_config.min_segment_num,
            complementary_segment=flashvid_config.complementary_segment,
        )
    else:
        segment_lengths = torch.tensor([num_frames], dtype=torch.long, device=video_features.device)

    num_segments = segment_lengths.shape[0]
    global_indices = torch.arange(num_frames * num_visual_tokens, dtype=torch.long, device=video_features.device)

    token_budget = math.ceil(num_visual_tokens * flashvid_config.retention_ratio * flashvid_config.expansion)
    num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
    num_sttm_tokens = token_budget - num_attn_div_tokens
    flashvid_config.num_attn_div_tokens = num_attn_div_tokens
    flashvid_config.num_sttm_tokens = num_sttm_tokens

    all_segment_features, all_segment_indices, all_segment_sizes = [], [], []
    expected_mass = 0
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = segment_lengths[seg_idx]
        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        segment_features, segment_global_indices, segment_sizes, segment_expected = _segment_compression_with_mass(
            segment_features=segment_features,
            segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention,
            flashvid_config=flashvid_config,
        )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        all_segment_sizes.append(segment_sizes)
        expected_mass += segment_expected
        offset += seg_len
    final_tokens = torch.cat(all_segment_features, dim=0)
    final_global_indices = torch.cat(all_segment_indices, dim=0)
    final_sizes = torch.cat(all_segment_sizes, dim=0)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]
    flashvid_config.visual_token_length = sorted_tokens.shape[0]

    total = int(final_sizes.sum())
    if total != expected_mass:
        raise AssertionError(
            f"fvmass bookkeeping: masses sum to {total}, expected {expected_mass} "
            f"for {num_frames} frames x {num_visual_tokens} tokens")
    if getattr(flashvid_config, "mass", True):
        flashvid_config.token_mass = final_sizes[sorted_indices]
    return sorted_tokens, final_global_indices[sorted_indices]


def _segment_compression_with_mass(segment_features, segment_global_indices, cls_attention, flashvid_config):
    """``flashvid.utils.segment_compression`` plus per-token group sizes.

    Returns ``(tokens, global_indices, sizes, expected)``, where ``expected`` is
    what the sizes must add up to on the path this segment took: every input
    token when TSTM runs (plus the ADTS tokens a single-frame segment delivers a
    second time), only the ADTS tokens when it does not, because upstream then
    delivers nothing else.
    """
    num_frames, num_visual_tokens, feat_dim = segment_features.shape
    device = segment_features.device

    if flashvid_config.alpha > 0:
        additional_kwargs = {"cls_attention": cls_attention} if "attn" in flashvid_config.token_selection_method else {}
        selected_features, selected_indices = ALL_TOKEN_SELECTION_METHOD[flashvid_config.token_selection_method](
            features=segment_features,
            num_retained_tokens=flashvid_config.num_attn_div_tokens,
            **additional_kwargs,
        )
        selected_global_indices = segment_global_indices.gather(1, index=selected_indices).view(-1)
    else:
        selected_features = torch.tensor([]).to(segment_features)
        selected_indices = torch.tensor([]).to(segment_global_indices)
        selected_global_indices = torch.tensor([]).to(segment_global_indices)

    mask = torch.ones(num_frames, num_visual_tokens, dtype=torch.bool, device=device)
    mask.scatter_(1, selected_indices, False)

    expected = int(selected_global_indices.numel())
    num_other_tokens = flashvid_config.num_sttm_tokens * num_frames
    if num_other_tokens > 0 and flashvid_config.temporal_threshold < 1.0:
        expected = num_frames * num_visual_tokens
        if num_frames > 1:
            temp_merged_token_list, temp_merged_indices_list, temp_merged_size_list = _spatiotemporal_compression_with_mass(
                video_features=segment_features,
                temporal_threshold=flashvid_config.temporal_threshold,
                token_mask=mask,
                flashvid_config=flashvid_config,
            )
            temp_merged_global_indices_list = [segment_global_indices.view(num_frames, -1)[i][temp_merged_indices] for i, temp_merged_indices in enumerate(temp_merged_indices_list)]
        else:
            # Upstream passes every token of the frame on, the ADTS ones included.
            temp_merged_token_list = [segment_features[0]]
            temp_merged_global_indices_list = [segment_global_indices[0]]
            temp_merged_size_list = [torch.ones(num_visual_tokens, dtype=torch.long, device=device)]
            expected += int(selected_global_indices.numel())
    else:
        temp_merged_token_list = []
        temp_merged_global_indices_list = []
        temp_merged_size_list = []

    all_tokens = [selected_features.view(-1, feat_dim)]
    all_global_indices = [selected_global_indices]
    all_sizes = [torch.ones(selected_global_indices.numel(), dtype=torch.long, device=device)]
    if num_other_tokens > 0:
        num_current_retained_tokens = sum(len(tokens) for tokens in temp_merged_token_list)
        adapative_contextual_ratio = num_other_tokens / num_current_retained_tokens
        if adapative_contextual_ratio < 1.0:
            num_frames_in_segment = len(temp_merged_token_list)
            max_num_tokens = max(len(tokens) for tokens in temp_merged_token_list)
            batched_tokens = torch.zeros((num_frames_in_segment, max_num_tokens, feat_dim), dtype=segment_features.dtype, device=device)
            valid_token_mask = torch.zeros((num_frames_in_segment, max_num_tokens), dtype=torch.bool, device=device)
            num_clusters_list = []
            k_list = []
            for i, temp_merged_tokens in enumerate(temp_merged_token_list):
                num_tokens = len(temp_merged_tokens)
                batched_tokens[i, :num_tokens] = temp_merged_tokens
                valid_token_mask[i, :num_tokens] = True
                num_clusters = math.ceil(num_tokens * adapative_contextual_ratio)
                num_clusters_list.append(num_clusters)
                k_list.append(min(num_clusters, 7))
            cluster_indices_list, cluster_center_indices_list = dpc_knn(
                features=batched_tokens,
                num_clusters=num_clusters_list,
                k=k_list,
                valid_token_mask=valid_token_mask,
            )
            for i, (temp_merged_tokens, temp_merged_global_indices, temp_merged_sizes) in enumerate(
                    zip(temp_merged_token_list, temp_merged_global_indices_list, temp_merged_size_list)):
                num_clusters = num_clusters_list[i]
                if num_clusters > 0:
                    cluster_indices = cluster_indices_list[i][:len(temp_merged_tokens)]
                    cluster_center_indices = cluster_center_indices_list[i]
                    aggregated_tokens = torch.zeros((num_clusters, feat_dim), dtype=segment_features.dtype, device=device)
                    aggregated_tokens.scatter_add_(0, cluster_indices.unsqueeze(-1).expand(-1, feat_dim), temp_merged_tokens)
                    cluster_counts = torch.bincount(cluster_indices, minlength=num_clusters).unsqueeze(-1).to(segment_features.dtype)
                    aggregated_tokens = aggregated_tokens / cluster_counts
                    global_token_indices = temp_merged_global_indices[cluster_center_indices]
                    sizes = torch.zeros(num_clusters, dtype=torch.long, device=device)
                    sizes.scatter_add_(0, cluster_indices, temp_merged_sizes)
                else:
                    aggregated_tokens = temp_merged_tokens
                    global_token_indices = temp_merged_global_indices
                    sizes = temp_merged_sizes

                all_tokens.append(aggregated_tokens)
                all_global_indices.append(global_token_indices)
                all_sizes.append(sizes)
        else:
            for temp_merged_tokens, temp_merged_global_indices, temp_merged_sizes in zip(
                    temp_merged_token_list, temp_merged_global_indices_list, temp_merged_size_list):
                all_tokens.append(temp_merged_tokens)
                all_global_indices.append(temp_merged_global_indices)
                all_sizes.append(temp_merged_sizes)

    return torch.cat(all_tokens, dim=0), torch.cat(all_global_indices, dim=0), torch.cat(all_sizes, dim=0), expected


def _spatiotemporal_compression_with_mass(video_features, temporal_threshold, token_mask, flashvid_config):
    """``flashvid.utils.spatiotemporal_compression`` plus the size of each retained tree."""
    num_frames, num_visual_tokens, feat_dim = video_features.shape
    lower_bound = (flashvid_config.num_attn_div_tokens + flashvid_config.num_sttm_tokens) * num_frames
    normed_video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
    cosine_similarities = torch.bmm(normed_video_features[1:], normed_video_features[:-1].transpose(1, 2))
    cosine_similarities[~token_mask[1:].unsqueeze(-1).expand(-1, -1, num_visual_tokens)] = -1.0
    cosine_similarities[~token_mask[:-1].unsqueeze(1).expand(-1, num_visual_tokens, -1)] = -1.0

    max_sims, max_sim_indices = torch.max(cosine_similarities, dim=-1)

    padded_max_sims = F.pad(max_sims, (0, 0, 1, 0), value=-1)
    padded_max_sim_indices = F.pad(max_sim_indices, (0, 0, 1, 0), value=-1)

    token_counts = torch.ones(num_frames, num_visual_tokens).to(video_features)
    tree_sizes = torch.ones(num_frames, num_visual_tokens, dtype=torch.long, device=video_features.device)
    mask = padded_max_sims > temporal_threshold
    retaining_token_mask = ~mask

    if retaining_token_mask.int().sum() < lower_bound:
        soft_threshold = padded_max_sims.view(-1).topk(k=(num_frames * num_visual_tokens) - lower_bound).values[-1]
        soft_threshold = max(soft_threshold, -1.0 + 1e-6)
        mask = padded_max_sims > soft_threshold
        retaining_token_mask = ~mask

    for frame_idx in range(num_frames - 1, -1, -1):
        frame_features = video_features[frame_idx]
        frame_token_counts = token_counts[frame_idx]
        frame_max_sim_indices = padded_max_sim_indices[frame_idx]

        tokens_to_merge = frame_features[~mask[frame_idx]]
        to_merge_token_counts = frame_token_counts[~mask[frame_idx]]
        if tokens_to_merge.numel() > 0:
            aggregated_tokens = tokens_to_merge / to_merge_token_counts.unsqueeze(-1).to(tokens_to_merge.dtype)
            video_features[frame_idx][~mask[frame_idx]] = aggregated_tokens
            token_counts[frame_idx][~mask[frame_idx]] = 1

        other_tokens = frame_features[mask[frame_idx]]
        if other_tokens.numel() > 0:
            anchor_token_indices = frame_max_sim_indices[mask[frame_idx]]
            aggregated_tokens = torch.zeros((num_visual_tokens, feat_dim), dtype=video_features.dtype, device=video_features.device)
            aggregated_tokens.scatter_add_(0, anchor_token_indices.unsqueeze(-1).expand(-1, feat_dim), other_tokens)
            aggregated_token_counts = torch.bincount(anchor_token_indices, minlength=num_visual_tokens).to(video_features.dtype)
            video_features[frame_idx - 1] += aggregated_tokens
            token_counts[frame_idx - 1] += aggregated_token_counts
            token_counts[frame_idx][mask[frame_idx]] = 0
            # A tree moves to its anchor whole: sizes accumulate and are never reset.
            tree_sizes[frame_idx - 1].scatter_add_(0, anchor_token_indices, tree_sizes[frame_idx][mask[frame_idx]])
            tree_sizes[frame_idx][mask[frame_idx]] = 0

    final_tokens, retained_token_indices, retained_sizes = [], [], []
    for i in range(num_frames):
        frame_mask = retaining_token_mask[i] & token_mask[i]
        final_tokens.append(video_features[i][frame_mask])
        retained_token_indices.append(torch.where(frame_mask)[0])
        retained_sizes.append(tree_sizes[i][frame_mask])

    return final_tokens, retained_token_indices, retained_sizes


def fvmass_prune(hidden_states, causal_mask, attentions, cache_position, position_ids,
                 position_embeddings, flashvid_config, visual_pos_masks=None):
    """FlashVID's ``fastv_prune``, with the mass vector and the bias kept on the kept keys."""
    flashvid_config.fvmass_pruned = True
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
    m = getattr(flashvid_config, "token_mass", None)
    if m is None:
        return out
    hidden_states, causal_mask, position_ids, cache_position, position_embeddings, keep_indices = out

    in_span = (keep_indices >= start) & (keep_indices < start + n_before)
    m_kept = m.to(keep_indices.device)[keep_indices[in_span] - start]
    if m_kept.numel() != int(flashvid_config.visual_token_length):
        raise AssertionError(
            f"fvmass prune: kept {m_kept.numel()} masses for {flashvid_config.visual_token_length} visual tokens")
    flashvid_config.token_mass = m_kept

    if causal_mask is None or causal_mask.dtype == torch.bool:
        raise AssertionError("fvmass prune: expected the additive float mask the prefill bias built")
    floor = torch.finfo(causal_mask.dtype).min
    rebuilt = torch.zeros_like(causal_mask)
    rebuilt.masked_fill_(causal_mask <= floor / 2, floor)
    beta = m_kept.to(torch.float32).clamp(min=1.0).log().to(causal_mask.dtype)
    rebuilt[..., start:start + m_kept.numel()] += beta
    return hidden_states, rebuilt, position_ids, cache_position, position_embeddings, keep_indices


def fvmass_score_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values=None):
    """``apply_mass_bias`` in prefill, and in decode unless inner pruning ran (see module docstring)."""
    if hidden_states.shape[1] > 1:
        return apply_mass_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values)
    pruned = getattr(flashvid_config, "fvmass_pruned", False)
    if not getattr(flashvid_config, "_fvmass_decode_logged", False):
        # Once per process: which of the two decode paths this run is on.
        flashvid_config._fvmass_decode_logged = True
        on = not pruned and getattr(flashvid_config, "token_mass", None) is not None
        print(f"[BV] fvmass decode: inner pruning ran {pruned}, log-mass bias {'on' if on else 'off'}", flush=True)
    if pruned:
        return causal_mask
    return apply_mass_bias(causal_mask, hidden_states, cache_position, flashvid_config, past_key_values)
