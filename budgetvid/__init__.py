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

Everything above needs FlashVID, which needs transformers and a CUDA-ish
environment. ``budgetvid.core`` deliberately needs neither, so that the scoring
and routing arithmetic can be unit-tested on a laptop while the GPU is busy.
Importing this package must therefore not, by itself, drag transformers in --
hence the PEP 562 lazy attribute access below. Touching ``budgetvid`` or
``BudgetVidConfig`` loads the heavy half and performs the dispatch
registrations, exactly as an eager import used to.
"""

from __future__ import annotations  # `nn` is only bound after _load_heavy()

from dataclasses import asdict

__all__ = [
    "budgetvid",
    "BudgetVidConfig",
    "register_allocation",
    "available_allocations",
]

_HEAVY = {"BudgetVidConfig", "register_allocation", "available_allocations"}


def _load_heavy():
    """Import the FlashVID-dependent half and perform dispatch registration."""
    global nn, _apply_flashvid, register_compression, register_llm_pruning
    global fastv_prune, available_allocations, register_allocation
    global budgetvid_compression, BudgetVidConfig, _loaded

    from torch import nn  # noqa: F811

    from flashvid import flashvid as _apply_flashvid  # noqa: F811
    from flashvid.dispatch import register_compression, register_llm_pruning  # noqa: F811
    from flashvid.utils import fastv_prune  # noqa: F811

    from .allocation import available_allocations, register_allocation  # noqa: F811
    from .compression import budgetvid_compression  # noqa: F811
    from .configuration_budgetvid import BudgetVidConfig  # noqa: F811
    from .adapters.pipeline import budgetvid_pipeline  # noqa: F811

    register_compression("budgetvid", budgetvid_compression)
    # The method this project's own experiments run under. Separate from the
    # allocation-based "budgetvid" entry so the two cannot be confused in a
    # results table.
    register_compression("bv", budgetvid_pipeline)
    # The inner-LLM stage is FlashVID's for now. Note that it is a *second*,
    # independent budget: the vision side keeps `retention_ratio * expansion` of
    # the tokens, then layer `pruning_layer` cuts to `llm_retention_ratio` of
    # what is left. Making the two budgets one allocation is an open thread, and
    # the place to do it is here.
    register_llm_pruning("budgetvid", fastv_prune)
    _loaded = True


_loaded = False


def __getattr__(name):
    # Only consulted for names that are NOT already module-level, which is why
    # `budgetvid` itself is absent from _HEAVY -- it is a real function below and
    # calls _load_heavy() on entry instead.
    if name in _HEAVY:
        if not _loaded:
            _load_heavy()
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _retarget_config(model, config) -> int:
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


def budgetvid(model: nn.Module, allocation: str = "uniform", enforce_budget: bool = True,
              policy: str | None = None, seed: int = 42, **flashvid_kwargs) -> nn.Module:
    """Apply BudgetVID to the model.

    Args:
        model (nn.Module): The model to patch. Same models ``flashvid()``
            supports: LLaVA-OneVision, LLaVA-Video, Qwen2.5-VL, Qwen3-VL.
        allocation (str, optional): Name of the budget allocation policy, see
            ``budgetvid/allocation.py``. Defaults to "uniform", which
            reproduces FlashVID.
        enforce_budget (bool, optional): Raise if a policy spends more than the
            global budget. Defaults to True.
        policy (str, optional): When given, route vision-side compression to
            ``budgetvid/adapters/pipeline.py`` (method ``bv``) and run this
            policy -- "none", "random_drop", "uniform". Leaving it None keeps
            the allocation-based path.
        seed (int, optional): Seed for policies with a random component.
        **flashvid_kwargs: Forwarded verbatim to ``flashvid()``
            (``retention_ratio``, ``alpha``, ``temporal_threshold``, ...).

    Raises:
        NotImplementedError: If ``flashvid()`` does not support the model.
        RuntimeError: If no config could be retargeted, which would mean
            ``flashvid()`` changed where it attaches.

    Returns:
        nn.Module: The patched model.
    """
    if not _loaded:
        _load_heavy()

    # Reuse flashvid()'s monkeypatching wholesale; it installs the shared
    # forwards and attaches a FlashVidConfig built from `flashvid_kwargs`.
    model = _apply_flashvid(model, **flashvid_kwargs)

    # Widen that config rather than rebuilding it, so a new upstream parameter
    # flows through without being listed again here.
    config = BudgetVidConfig(
        **asdict(model.flashvid_config),
        allocation=allocation,
        enforce_budget=enforce_budget,
        policy=policy or "none",
        seed=seed,
    )
    if policy is not None:
        # Route to budgetvid/adapters/pipeline.py rather than the allocation path.
        config.method = "bv"

    replaced = _retarget_config(model, config)
    if replaced == 0:
        raise RuntimeError("BudgetVID found no FlashVID config to retarget. flashvid() likely changed where it attaches its config; check flashvid/__init__.py.")

    return model
