from dataclasses import dataclass, field

from flashvid.configuration_flashvid import FlashVidConfig


@dataclass
class BudgetVidConfig(FlashVidConfig):
    """FlashVID's config plus the fields BudgetVID's allocation stage needs.

    Subclassing keeps every ``flashvid.utils`` helper usable unchanged: they
    only ever read the inherited fields, and they type-hint ``FlashVidConfig``.
    """

    # Selects the policy in `flashvid.dispatch`; see `budgetvid/__init__.py`.
    method: str = field(default="bv")


    # Vision-side policy for the `bv` method, see `budgetvid/adapters/pipeline.py`.
    # Every policy shares one assembly path, which is what keeps an ablation row
    # comparable to a baseline row.
    policy: str = field(default="none")



    # --- BudgetVID 2.0 (measure quantization), spec 2026-08-28_method_budgetvid2_v1 ---
    # Metric space the grouping decisions are taken in (spec eq 1):
    #   "kv"   -> (W_k RN(x), sqrt(gamma_v) W_v RN(x)), the default
    #   "key"  -> key half only (gamma_v = 0)
    #   "none" -> the projector space itself, the metric ablation
    lift: str = field(default="kv")
    gamma_v: float = field(default=1.0)
    # RMSNorm inside the lift. On by default because the key the decoder forms
    # is W_k RN(x) and never W_k x; off is the pre-freeze norm-free variant.
    lift_norm: bool = field(default=True)
    # "waterfill" (CBA, spec eq 3) or "even" (v0's largest remainder), which is
    # the allocation ablation and must reproduce v0's split exactly. "video"
    # drops the per-frame split: one FPS over all L*N_f tokens, one-seed floor
    # per frame (notes/2026-09-24_note_video-level-coreset-design.html).
    mq_alloc: str = field(default="waterfill")
    # "rms" delivers the metric centroid's direction at the group's mean token
    # norm (L1'); "plain" is the unweighted mean every prior merge uses.
    centroid: str = field(default="rms")
    # Cap on the cost-curve length; 0 means N_f (exact curves).
    b_max: int = field(default=0)
    # Lloyd sweeps after the FPS seeding. 0 is the frozen v1 behaviour; >0 moves
    # each center to its group's mean, which is what stops one seed absorbing a
    # frame's dense bulk (see core/quantize.py:lloyd_refine).
    refine: int = field(default=0)
    # The mass channel (spec eq 5). False is the mandatory ablation: a
    # conventional mass-destroying merge. Requires an attention implementation
    # that accepts an additive mask -- sdpa or eager, never flash_attention_2.
    mass: bool = field(default=True)
    # Inner-LLM stage for method bv: "none" (one budget, every layer sees B
    # tokens) or "fastv" (FlashVID's second stage, unchanged: at `pruning_layer`
    # keep `llm_retention_ratio` of the visual tokens by last-query attention).
    # See budgetvid/llm_prune.py.
    llm_prune: str = field(default="none")
    # Set by budgetvid() when the decoder is moved to sdpa for the mass bias.
    # Recorded rather than assumed, so a run's metadata says which backend the
    # LANGUAGE model actually used (the vision tower stays on FA2 either way).
    attn_text: str = field(default="")


    # Per-video signal/routing dumps (budgetvid/recording.py). Empty = off.
    # The per-sample tag is set by the eval wrapper (`dump_tag`, a transient
    # attribute, deliberately not a dataclass field: it changes every sample).
    dump_dir: str = field(default="")
