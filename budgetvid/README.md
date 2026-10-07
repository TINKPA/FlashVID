# BudgetVID / MassVID

Budget-aware token compression, built on FlashVID. The method is **MassVID**
(measure quantization, spec frozen at `notes/2026-08-28_method_budgetvid2_v1.html`):
`policy=mq`. The Round-1 code (score -> route -> merge, uniform / random_drop /
attn_top baselines, the allocation-based `budgetvid` method) was removed on
2026-10-07; commit 89c59fe is the last one that carries it, and every run that
used it records its commit.

## Where things live

| Path | What it is |
|---|---|
| `core/quantize.py` | Metric lift, FPS cost curves, water-filling, MPQ, Lloyd refine, medoid delivery, video-level coreset. |
| `core/assembly.py`, `core/budget.py` | Assemble the compressed sequence (with masses); even split used by the `even` ablation and the dumps. |
| `mass_bias.py` | `beta = log m` as an additive attention-score bias. |
| `adapters/pipeline.py` | Method `bv`: `policy=none` (vanilla) or `policy=mq`; inner-LLM stage is a deliberate no-op. |
| `flashvid_mass.py` | Method `fvmass`: FlashVID's own compression + FastV pruning, with the mass channel (`policy=flashvid_mass`). |
| `recording.py` | Per-video dumps (`dump_dir=`). |
| `configuration_budgetvid.py` | `FlashVidConfig` plus every knob below. |
| `__init__.py` | The `budgetvid()` wrapper and dispatch registration. |

## Running

```bash
# the method
--model_args ...,enable_budgetvid=True,policy=mq,retention_ratio=0.125

# its ablations, one flag each
mass=False        # the mass channel off: a conventional mass-destroying merge
mq_alloc=even     # largest-remainder split instead of water-filling (CBA off)
mq_alloc=video    # one FPS over the whole video, one-seed floor per frame
lift=none         # measure in projector space instead of the decoder's key space
lift_norm=False   # drop the RMSNorm from the lift
centroid=plain    # unweighted mean instead of the metric centroid direction
centroid=medoid   # deliver a real token (weighted coreset)
refine=5          # Lloyd sweeps after FPS seeding
```

`retention_ratio` here is the fraction of visual tokens at the LLM input,
B = round(r * L * N_f), held at every layer. FlashVID's `retention_ratio` is a
per-layer average (vision side keeps `R * expansion`, FastV at `pruning_layer`
keeps `llm_retention_ratio` of that), so FlashVID R sits at 1.25 R on this axis.

**Two attention backends, on purpose.** The bias rides on an additive attention
mask and flash_attention_2 does not take one, so `budgetvid()` moves the decoder
to sdpa whenever `mass=True`. It moves *only* the decoder: the vision tower
asserts FA2 because the varlen `cu_seqlens` path is how the [CLS] attention is
extracted. `text_sdpa=True` performs the switch on its own, for a backend control.

## Tests (no GPU)

```bash
for t in budgetvid/tests/test_mq.py budgetvid/tests/test_mq_pipeline.py \
         budgetvid/tests/test_mq_video.py budgetvid/tests/test_flashvid_mass.py; do
  uv run --no-project --python 3.11 --with torch --with numpy python $t
done
```

Three things fail silently rather than loudly, so they have tests: the mass
vector must travel through `assemble`'s sort **with** the tokens; a group of one
must come back out as the identity; `mq_alloc=even` must reproduce the
largest-remainder split exactly.

Everything else is FlashVID's, imported not copied. The only edits to upstream
files are the call sites routed through `flashvid/dispatch.py`. Keep it that
way; every extra line there is a line to merge by hand on the next rebase.
