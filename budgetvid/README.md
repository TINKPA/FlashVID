# BudgetVID

Budget-aware token compression, built on FlashVID.

Two methods live here. **2.0 (measure quantization)** is the current one, spec
frozen at `notes/2026-08-28_method_budgetvid2_v1.html`; **version 0** (score
`S = I_hat - lambda*R_hat`, then route to retain/merge/drop) stays as the
baseline and ablation reference. Both run under `policy=` on the same assembly
path, which is the only reason a 2.0 row and a v0 row are comparable at all.

FlashVID spends the same per-frame token budget on every segment of a video
(`flashvid/utils.py:46`), so a static shot and a busy one get compressed equally
hard. BudgetVID routes that decision through an allocation policy instead.

## Where things live

| Path | What it is |
|---|---|
| `core/quantize.py` | **2.0.** Metric lift, FPS cost curves, water-filling, MPQ. |
| `mass_bias.py` | **2.0.** `beta = log m` as an additive attention-score bias. |
| `adapters/pipeline.py` | Method `bv`: one policy switch, one assembly path. |
| `core/scoring.py`, `core/routing.py`, `core/merging.py` | version 0. |
| `allocation.py` | Budget allocation policies for the older `budgetvid` method. |
| `compression.py` | FlashVID's vision-side stage with a per-segment budget. |
| `configuration_budgetvid.py` | `FlashVidConfig` plus every knob below. |
| `__init__.py` | The `budgetvid()` wrapper and dispatch registration. |

## Running 2.0

```bash
# the method
--model_args ...,attn_implementation=sdpa,enable_budgetvid=True,policy=mq,retention_ratio=0.0625

# its ablations, one flag each
mass=False        # the mass channel off: a conventional mass-destroying merge
mq_alloc=even     # v0's largest-remainder split instead of water-filling (CBA off)
lift=none         # measure in projector space instead of the decoder's key space
lift_norm=False   # drop the RMSNorm from the lift
centroid=plain    # unweighted mean instead of the metric centroid direction
```

`attn_implementation=sdpa` is **required** while `mass=True`: the bias rides on
an additive attention mask and flash_attention_2 does not take one. Passing FA2
raises at patch time rather than silently producing an unbiased -- and therefore
wrong -- number.

Three things about 2.0 fail silently rather than loudly, so they have tests
(`tests/test_mq.py`, `tests/test_mq_pipeline.py`, 48 checks, no GPU):

* the mass vector must travel through `assemble`'s sort **with** the tokens, or
  the bias lands on the wrong keys;
* the delivered token's *direction* comes from the metric centroid but its
  *magnitude* from the group's mean token norm -- the residual stream carries
  the vector itself, so a group of one has to come back out as the identity;
* `mq_alloc=even` must reproduce v0's split exactly, which is the regression
  check on the allocation plumbing.

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
