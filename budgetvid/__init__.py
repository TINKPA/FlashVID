"""BudgetVID: budget-aware token pruning for video LLMs.

BudgetVID reuses FlashVID's scaffolding wholesale. The monkeypatched
``forward`` bodies, DySeg, ADTS, TSTM and the inner-LLM pruning stage are all
FlashVID's code, imported not copied, so an upstream rebase carries straight
over and ``flashvid()`` stays available untouched as a baseline.

What BudgetVID changes is one thing: FlashVID spends the same per-frame token
budget on every segment of a video, so a static shot and a busy one are
compressed equally hard. BudgetVID routes that decision through an allocation
policy instead.

Usage mirrors ``flashvid()``::

    from budgetvid import budgetvid

    model = budgetvid(model, allocation="uniform", retention_ratio=0.25)

``allocation="uniform"`` reproduces FlashVID exactly and exists as the
regression check. Write new policies in ``budgetvid/allocation.py``.
"""

from dataclasses import asdict

from torch import nn

from flashvid import flashvid as _apply_flashvid
from flashvid.dispatch import register_compression, register_llm_pruning
from flashvid.utils import fastv_prune

from .allocation import available_allocations, register_allocation
from .compression import budgetvid_compression
from .configuration_budgetvid import BudgetVidConfig

__all__ = [
    "budgetvid",
    "BudgetVidConfig",
    "register_allocation",
    "available_allocations",
]

register_compression("budgetvid", budgetvid_compression)
# The inner-LLM stage is FlashVID's for now. Note that it is a *second*,
# independent budget: the vision side keeps `retention_ratio * expansion` of the
# tokens, then layer `pruning_layer` cuts to `llm_retention_ratio` of what is
# left. Making the two budgets one allocation is an open thread, and the place
# to do it is here.
register_llm_pruning("budgetvid", fastv_prune)


def _retarget_config(model: nn.Module, config: BudgetVidConfig) -> int:
    """Swap BudgetVID's config in wherever ``flashvid()`` attached FlashVID's.

    ``flashvid()`` attaches its config to a model-type-dependent set of modules.
    Rather than restate that list, which would go stale the moment upstream adds
    a target, this walks the module tree and replaces every config it finds.

    Args:
        model (nn.Module): A model that ``flashvid()`` has already patched.
        config (BudgetVidConfig): The config to install.

    Returns:
        int: How many modules were retargeted.
    """
    replaced = 0
    for module in model.modules():
        if getattr(module, "flashvid_config", None) is not None:
            module.flashvid_config = config
            replaced += 1
    return replaced


def budgetvid(model: nn.Module, allocation: str = "uniform", enforce_budget: bool = True, **flashvid_kwargs) -> nn.Module:
    """Apply BudgetVID to the model.

    Args:
        model (nn.Module): The model to patch. Same models ``flashvid()``
            supports: LLaVA-OneVision, LLaVA-Video, Qwen2.5-VL, Qwen3-VL.
        allocation (str, optional): Name of the budget allocation policy, see
            ``budgetvid/allocation.py``. Defaults to "uniform", which
            reproduces FlashVID.
        enforce_budget (bool, optional): Raise if a policy spends more than the
            global budget. Defaults to True.
        **flashvid_kwargs: Forwarded verbatim to ``flashvid()``
            (``retention_ratio``, ``alpha``, ``temporal_threshold``, ...).

    Raises:
        NotImplementedError: If ``flashvid()`` does not support the model.
        RuntimeError: If no config could be retargeted, which would mean
            ``flashvid()`` changed where it attaches.

    Returns:
        nn.Module: The patched model.
    """
    # Reuse flashvid()'s monkeypatching wholesale; it installs the shared
    # forwards and attaches a FlashVidConfig built from `flashvid_kwargs`.
    model = _apply_flashvid(model, **flashvid_kwargs)

    # Widen that config rather than rebuilding it, so a new upstream parameter
    # flows through without being listed again here.
    config = BudgetVidConfig(
        **asdict(model.flashvid_config),
        allocation=allocation,
        enforce_budget=enforce_budget,
    )

    replaced = _retarget_config(model, config)
    if replaced == 0:
        raise RuntimeError("BudgetVID found no FlashVID config to retarget. flashvid() likely changed where it attaches its config; check flashvid/__init__.py.")

    return model
