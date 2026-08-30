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
    global register_score_bias, apply_mass_bias
    global fastv_prune, available_allocations, register_allocation
    global budgetvid_compression, BudgetVidConfig, _loaded

    from torch import nn  # noqa: F811

    from flashvid import flashvid as _apply_flashvid  # noqa: F811
    from flashvid.dispatch import (  # noqa: F811
        register_compression, register_llm_pruning, register_score_bias)
    from flashvid.utils import fastv_prune  # noqa: F811

    from .allocation import available_allocations, register_allocation  # noqa: F811
    from .mass_bias import apply_mass_bias  # noqa: F811
    from .compression import budgetvid_compression  # noqa: F811
    from .configuration_budgetvid import BudgetVidConfig  # noqa: F811
    from .adapters.pipeline import budgetvid_pipeline, no_llm_pruning  # noqa: F811

    register_compression("budgetvid", budgetvid_compression)
    # The method this project's own experiments run under. Separate from the
    # allocation-based "budgetvid" entry so the two cannot be confused in a
    # results table.
    register_compression("bv", budgetvid_pipeline)
    # `bv` has a single budget by construction, so the inner-LLM stage is a
    # no-op. Without this the dispatch raises KeyError at `pruning_layer`.
    register_llm_pruning("bv", no_llm_pruning)
    # The mass channel of BudgetVID 2.0 (spec eq 5). A no-op for every policy
    # that leaves `token_mass` unset, so the other rows are untouched.
    register_score_bias("bv", apply_mass_bias)
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


def _capture_lift(model) -> dict | None:
    """Grab the frozen tensors the 2.0 metric lift measures distances through.

    The decoder's FIRST layer is the one whose attention read the guarantee is
    stated for, so this walks the module tree and takes the first decoder layer
    it finds that exposes ``self_attn.k_proj`` and ``input_layernorm``. That is
    the text stack on every model ``flashvid()`` supports -- the vision blocks
    of Qwen2.5-VL/Qwen3-VL fuse their projections into one ``qkv``, so they
    cannot match by accident.

    References, not copies: the tensors follow the model's device and dtype, and
    nothing here is ever written to.

    Returns:
        ``{"W_k", "W_v", "g", "eps"}`` or None when no such layer exists (which
        is fine unless a policy asks for a lift, where it becomes an error).
    """
    for module in model.modules():
        attn = getattr(module, "self_attn", None)
        ln = getattr(module, "input_layernorm", None)
        if attn is None or ln is None or not hasattr(attn, "k_proj"):
            continue
        return {
            "W_k": attn.k_proj.weight.detach(),
            "W_v": attn.v_proj.weight.detach(),
            "g": ln.weight.detach(),
            "eps": float(getattr(ln, "variance_epsilon", getattr(ln, "eps", 1e-6))),
        }
    return None


def _text_stack_to_sdpa(model) -> int:
    """Put the LANGUAGE model on sdpa, leaving the vision tower on FA2.

    The log-mass bias is an additive attention-score bias, and
    flash_attention_2 takes no additive mask -- but the fork's vision tower
    *requires* FA2: its patched attention asserts it outright
    (flashvid/modeling_qwen2_5_vl.py:560), because the varlen ``cu_seqlens``
    path is how it extracts the [CLS] attention that every policy scores with.
    Loading the whole model under sdpa therefore fails in the encoder, before
    compression is ever reached.

    The two are separable: transformers keeps one ``_attn_implementation`` per
    sub-config, read at forward time. Only the decoder needs the bias, so only
    the decoder is switched, and the encoder keeps the fast varlen kernel.

    Returns:
        int: how many config objects were switched; 0 means the text stack was
        not found, which the caller must treat as an error rather than run an
        unbiased -- and therefore wrong -- benchmark.
    """
    for module in model.modules():
        layers = getattr(module, "layers", None)
        if layers is None or len(layers) == 0:
            continue
        attn = getattr(layers[0], "self_attn", None)
        if attn is None or not hasattr(attn, "k_proj"):
            continue          # vision blocks fuse into .attn/qkv, so they can't match
        seen, n = set(), 0
        holders = [module, layers[0], attn] + [getattr(l, "self_attn", None) for l in layers]
        for holder in holders:
            cfg = getattr(holder, "config", None)
            if cfg is not None and id(cfg) not in seen:
                cfg._attn_implementation = "sdpa"
                seen.add(id(cfg))
                n += 1
        return n
    return 0


def budgetvid(model: nn.Module, allocation: str = "uniform", enforce_budget: bool = True,
              policy: str | None = None, seed: int = 42,
              eta: float = 0.5, lam: float = 1.0,
              alpha_min: float = 0.4, alpha_max: float = 0.8,
              active_frac: float = 0.6, alpha_flip: bool = False,
              force_alpha: float = -1.0, debias_pos: bool = False,
              lift: str = "kv", gamma_v: float = 1.0, lift_norm: bool = True,
              mq_alloc: str = "waterfill", centroid: str = "rms",
              b_max: int = 0, mass: bool = True, text_sdpa: bool = False,
              refine: int = 0,
              dump_dir: str = "",
              **flashvid_kwargs) -> nn.Module:
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
        lift (str, optional): Metric space for policy ``mq`` -- "kv", "key" or
            "none". See BudgetVidConfig for what each means.
        gamma_v (float, optional): Weight of the value half of the lift.
        lift_norm (bool, optional): Put the decoder's input RMSNorm inside the
            lift, which is the geometry the decoder actually reads.
        mq_alloc (str, optional): "waterfill" (CBA) or "even" (v0's split).
        centroid (str, optional): "rms" or "plain".
        b_max (int, optional): Cost-curve length cap; 0 means N_f.
        refine (int, optional): Lloyd sweeps after FPS seeding; 0 is the
            frozen v1 behaviour.
        mass (bool, optional): The log-mass attention bias. Turning it off is
            the mandatory ablation and needs no other change.
        text_sdpa (bool, optional): Move the decoder to sdpa even when nothing
            requires it. Implied by ``policy="mq"`` with ``mass=True``; set it
            by hand on a NON-2.0 policy to measure what the backend switch
            alone costs, which is the only way a 2.0 row can be compared to an
            older flash_attention_2 row with a number rather than a hope.
        dump_dir (str, optional): When set, every compressed video's signals
            and routing decisions are dumped there (budgetvid/recording.py).
            The eval wrapper sets ``dump_tag`` on the config per sample.
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
        eta=eta, lam=lam, alpha_min=alpha_min, alpha_max=alpha_max,
        active_frac=active_frac, alpha_flip=alpha_flip, force_alpha=force_alpha,
        debias_pos=debias_pos, dump_dir=dump_dir,
        lift=lift, gamma_v=gamma_v, lift_norm=lift_norm, mq_alloc=mq_alloc,
        centroid=centroid, b_max=b_max, mass=mass, refine=refine,
    )
    # Load the model under flash_attention_2 as usual -- the vision tower
    # demands it -- and move only the decoder to sdpa, which is what can carry
    # the additive log-mass bias. Failing loudly here beats discovering it as a
    # silently unbiased benchmark number.
    if text_sdpa or (policy == "mq" and mass):
        n = _text_stack_to_sdpa(model)
        if n == 0:
            raise RuntimeError(
                "the language model has to be on sdpa here (flash_attention_2 takes "
                "no additive score bias), but no text decoder stack was found to "
                "switch. Check budgetvid/__init__.py:_text_stack_to_sdpa against "
                "this model's layout.")
        config.attn_text = "sdpa"
        # Printed, not assumed: if this switch ever silently fails to happen,
        # the log-mass bias is silently not applied, and the mass row becomes
        # the ablation row wearing the method's name.
        print(f"[BV] decoder moved to sdpa for the log-mass bias "
              f"({n} config objects); vision tower stays on flash_attention_2",
              flush=True)
    if policy == "mq":
        config.lift_params = _capture_lift(model)
        config.token_mass = None
    if policy is not None:
        # Route to budgetvid/adapters/pipeline.py rather than the allocation path.
        config.method = "bv"

    replaced = _retarget_config(model, config)
    if replaced == 0:
        raise RuntimeError("BudgetVID found no FlashVID config to retarget. flashvid() likely changed where it attaches its config; check flashvid/__init__.py.")

    return model
