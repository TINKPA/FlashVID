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

    # Raise if a policy hands out more tokens than the global budget allows.
    # Overspending silently would invalidate every comparison against a
    # fixed-ratio baseline, so this defaults to on.
    enforce_budget: bool = field(default=True)
