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

    # Raise if a policy hands out more tokens than the global budget allows.
    # Overspending silently would invalidate every comparison against a
    # fixed-ratio baseline, so this defaults to on.
    enforce_budget: bool = field(default=True)

    # Per-video signal/routing dumps (budgetvid/recording.py). Empty = off.
    # The per-sample tag is set by the eval wrapper (`dump_tag`, a transient
    # attribute, deliberately not a dataclass field: it changes every sample).
    dump_dir: str = field(default="")
