from dataclasses import dataclass, field

from flashvid.configuration_flashvid import FlashVidConfig


@dataclass
class BudgetVidConfig(FlashVidConfig):
    """FlashVID's config plus the fields BudgetVID's allocation stage needs.

    Subclassing keeps every ``flashvid.utils`` helper usable unchanged: they
    only ever read the inherited fields, and they type-hint ``FlashVidConfig``.
    """

    # Selects the policy in `flashvid.dispatch`; see `budgetvid/__init__.py`.
    method: str = field(default="budgetvid")

    # Name of the budget allocation policy, see `budgetvid/allocation.py`.
    allocation: str = field(default="uniform")

    # Vision-side policy for the `bv` method, see `budgetvid/adapters/pipeline.py`.
    # Every policy shares one assembly path, which is what keeps an ablation row
    # comparable to a baseline row.
    policy: str = field(default="none")

    # Seed for any policy with a random component; kept off global RNG state so
    # benchmark numbers reproduce regardless of what the harness seeds.
    seed: int = field(default=42)

    # --- scoring / routing hyperparameters (spec §3.1-3.2) ---
    eta: float = field(default=0.5)          # R = (1-eta)*R_sp + eta*R_tp
    lam: float = field(default=1.0)          # S = I_hat - lam*R_hat
    alpha_min: float = field(default=0.4)
    alpha_max: float = field(default=0.8)
    # N_active/N_f stated directly. See core/routing.py for why beta cannot be
    # held fixed across retention ratios.
    active_frac: float = field(default=0.6)
    alpha_flip: bool = field(default=False)  # ablation row 5
    # Ablation row G: subtract the cross-video mean importance of each grid
    # position before ranking. Measures what the attention sink costs, without
    # making de-biasing part of the method (decision of 2026-08-04).
    debias_pos: bool = field(default=False)
    force_alpha: float = field(default=-1.0) # <0 means "unset"

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
    # the allocation ablation and must reproduce v0's split exactly.
    mq_alloc: str = field(default="waterfill")
    # "rms" delivers the metric centroid's direction at the group's mean token
    # norm (L1'); "plain" is the unweighted mean every prior merge uses.
    centroid: str = field(default="rms")
    # Cap on the cost-curve length; 0 means N_f (exact curves).
    b_max: int = field(default=0)
    # The mass channel (spec eq 5). False is the mandatory ablation: a
    # conventional mass-destroying merge. Requires an attention implementation
    # that accepts an additive mask -- sdpa or eager, never flash_attention_2.
    mass: bool = field(default=True)

    # Raise if a policy hands out more tokens than the global budget allows.
    # Overspending silently would invalidate every comparison against a
    # fixed-ratio baseline, so this defaults to on.
    enforce_budget: bool = field(default=True)

    # Per-video signal/routing dumps (budgetvid/recording.py). Empty = off.
    # The per-sample tag is set by the eval wrapper (`dump_tag`, a transient
    # attribute, deliberately not a dataclass field: it changes every sample).
    dump_dir: str = field(default="")
