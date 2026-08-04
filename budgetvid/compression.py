"""BudgetVID's vision-side compression stage.

This mirrors ``flashvid.utils.flashvid_compression`` with one change: instead of
computing a single per-frame token budget once and reusing it for every segment,
it asks an allocation policy how to distribute the global budget across
segments. Everything downstream (ADTS, TSTM, DPC-kNN) is FlashVID's code,
called unchanged, so ``allocation="uniform"`` is FlashVID exactly.

Keeping the per-segment stage as a call into ``flashvid.utils`` is deliberate:
until the intra-segment behaviour actually needs to change, there is nothing to
gain from a second copy that can drift. When it does need to change, copy
``segment_compression`` into this module and edit the copy.
"""

import math

import torch

from flashvid.utils import segment, segment_compression

from .allocation import allocate


def budgetvid_compression(video_features: torch.Tensor, cls_attention: torch.Tensor, flashvid_config):
    """Compress a video's visual tokens under a content-aware token budget.

    Args:
        video_features (torch.Tensor): Visual features, of shape
            (num_frames, num_visual_tokens, feat_dim).
        cls_attention (torch.Tensor): [CLS] attention, of shape
            (num_frames, num_visual_tokens).
        flashvid_config (BudgetVidConfig): The active config. Named for the
            argument ``flashvid.dispatch.compress`` passes, but it carries the
            BudgetVID fields.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Compressed tokens ordered by their
        position in the original sequence, and those global indices.
    """
    num_frames, num_visual_tokens, feat_dim = video_features.shape

    # 1. Partition the video into segments. FlashVID's DySeg, unchanged.
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

    # 2. Distribute the global token budget over those segments.
    #    This is the line FlashVID does not have: it spends
    #    `per_frame_budget` on every segment regardless of content.
    per_frame_budgets = allocate(
        video_features=video_features,
        cls_attention=cls_attention,
        segment_lengths=segment_lengths.tolist(),
        num_visual_tokens=num_visual_tokens,
        config=flashvid_config,
    )

    # 3. Compress each segment against its own budget. The split into ADTS and
    #    TSTM shares, and the compression itself, are FlashVID's.
    all_segment_features = []
    all_segment_indices = []
    offset = 0
    for seg_idx in range(num_segments):
        seg_len = segment_lengths[seg_idx]
        token_budget = per_frame_budgets[seg_idx]
        num_attn_div_tokens = math.ceil(token_budget * flashvid_config.alpha)
        # `segment_compression` reads both off the config, so they are set per
        # segment here rather than once before the loop.
        flashvid_config.num_attn_div_tokens = num_attn_div_tokens
        flashvid_config.num_sttm_tokens = token_budget - num_attn_div_tokens

        segment_features = video_features[offset : offset + seg_len]
        segment_cls_attention = cls_attention[offset : offset + seg_len]
        segment_global_indices = global_indices.view(num_frames, num_visual_tokens)[offset : offset + seg_len]
        segment_features, segment_global_indices = segment_compression(
            segment_features=segment_features,
            segment_global_indices=segment_global_indices,
            cls_attention=segment_cls_attention,
            flashvid_config=flashvid_config,
        )
        all_segment_features.append(segment_features)
        all_segment_indices.append(segment_global_indices)
        offset += seg_len

    final_tokens = torch.cat(all_segment_features, dim=0)
    final_global_indices = torch.cat(all_segment_indices, dim=0)

    sorted_indices = final_global_indices.argsort()
    sorted_tokens = final_tokens[sorted_indices]
    flashvid_config.visual_token_length = sorted_tokens.shape[0]
    return sorted_tokens, final_global_indices[sorted_indices]
