# BudgetVID

Budget-aware token pruning, built on FlashVID.

FlashVID spends the same per-frame token budget on every segment of a video
(`flashvid/utils.py:46`), so a static shot and a busy one get compressed equally
hard. BudgetVID routes that decision through an allocation policy instead.

## Where things live

| Path | What it is |
|---|---|
| `allocation.py` | **The research surface.** Budget allocation policies. |
| `compression.py` | FlashVID's vision-side stage with a per-segment budget. |
| `configuration_budgetvid.py` | `FlashVidConfig` plus `allocation` / `enforce_budget`. |
| `__init__.py` | The `budgetvid()` wrapper and policy registration. |

Everything else is FlashVID's, imported not copied: the monkeypatched `forward`
bodies, DySeg, ADTS, TSTM, DPC-kNN, and the inner-LLM pruning stage. Upstream
fixes to those carry over on rebase, and `flashvid()` stays available untouched
so it can serve as a baseline.

The only edits to upstream files are six call sites routed through
`flashvid/dispatch.py`, ten lines in total. Keep it that way; every extra line
there is a line to merge by hand on the next rebase.

## Writing a policy

```python
from budgetvid import register_allocation
from budgetvid.allocation import global_budget

def motion_weighted(video_features, cls_attention, segment_lengths, num_visual_tokens, config):
    """Give busy segments more tokens per frame than static ones."""
    ...
    return budgets  # one int per segment: tokens per frame in that segment

register_allocation("motion", motion_weighted)
```

A policy returns the per-frame budget for each segment, so segment `i` spends
`budgets[i] * segment_lengths[i]` tokens. The sum must stay at or below
`global_budget(...)`, which is exactly what FlashVID would spend at the same
`retention_ratio`. `allocate()` validates this on every call and raises if a
policy overspends, because a method that quietly uses more tokens than the
baseline is not being compared to it.

Policies must be deterministic, otherwise benchmark numbers stop reproducing.

## The regression check

`allocation="uniform"` reproduces FlashVID exactly. It must match
`enable_flashvid=True` to the decimal on every benchmark:

```bash
bash scripts/baseline/qwen2_5_vl.sh   # enable_flashvid=True
bash scripts/budgetvid/qwen2_5_vl.sh  # enable_budgetvid=True, allocation=uniform
```

If those two disagree, the plumbing is broken, not the idea. Check that before
trusting any number from a new policy.

The allocation arithmetic has unit tests that need no GPU:

```bash
uv run --no-project --python 3.11 python tests/test_allocation.py
```

## Open thread: there are two budgets

The vision side keeps `retention_ratio * expansion` of the tokens, then layer
`pruning_layer` cuts to `llm_retention_ratio` of what survived. Those are
independent knobs today, and `budgetvid` currently registers FlashVID's
`fastv_prune` unchanged for the second one. Making both stages draw on one
allocation is unexplored; the hook is `register_llm_pruning` in `__init__.py`.
