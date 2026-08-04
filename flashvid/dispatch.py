"""Method dispatch for the shared FlashVID scaffolding.

Upstream FlashVID hard-wires two policy calls into its monkeypatched forwards:

* vision-side compression -> ``flashvid.utils.flashvid_compression`` (3 sites)
* inner-LLM pruning -> ``flashvid.utils.fastv_prune`` (3 sites)

Those forwards (``modeling_qwen2.py``, ``modeling_qwen2_5_vl.py``,
``modeling_qwen3_vl.py``, ``llava_arch.py``, ``siglip_encoder.py``) are about
1800 lines of copied HuggingFace ``forward`` bodies that have to track
``transformers==4.57.3`` exactly, and upstream still patches them (e.g. the
2026-05-01 [CLS]-attention OOM fix). Forking them per method would mean redoing
that work on every rebase, so the six call sites go through this registry
instead and every method shares one copy of the scaffolding.

``flashvid`` is registered here and is the default, so behaviour is unchanged
when no other method has been imported. Other methods register themselves at
import time; see ``budgetvid/__init__.py``.
"""

from typing import Callable, Dict

from .configuration_flashvid import FlashVidConfig
from .utils import fastv_prune, flashvid_compression

DEFAULT_METHOD = "flashvid"

_COMPRESSION_REGISTRY: Dict[str, Callable] = {}
_LLM_PRUNING_REGISTRY: Dict[str, Callable] = {}


def register_compression(name: str, fn: Callable) -> None:
    """Register a vision-side compression policy under ``name``.

    Args:
        name (str): The method name, matched against ``config.method``.
        fn (Callable): Called as ``fn(video_features=..., cls_attention=...,
            flashvid_config=...)``, returning ``(tokens, global_indices)``.
    """
    _COMPRESSION_REGISTRY[name] = fn


def register_llm_pruning(name: str, fn: Callable) -> None:
    """Register an inner-LLM pruning policy under ``name``.

    Args:
        name (str): The method name, matched against ``config.method``.
        fn (Callable): Called with the same keyword arguments as
            ``flashvid.utils.fastv_prune``.
    """
    _LLM_PRUNING_REGISTRY[name] = fn


def _method_of(flashvid_config: FlashVidConfig) -> str:
    """Read the active method off the config, defaulting to plain FlashVID."""
    return getattr(flashvid_config, "method", DEFAULT_METHOD)


def _resolve(registry: Dict[str, Callable], method: str, stage: str) -> Callable:
    if method not in registry:
        raise KeyError(f"No {stage} policy registered for method '{method}'. " f"Registered: {sorted(registry)}. Did you forget to import the package that registers it?")
    return registry[method]


def compress(video_features, cls_attention, flashvid_config: FlashVidConfig):
    """Dispatch vision-side compression to the active method.

    Args:
        video_features (torch.Tensor): Visual features, of shape
            (num_frames, num_visual_tokens, feat_dim).
        cls_attention (torch.Tensor): [CLS] attention, of shape
            (num_frames, num_visual_tokens).
        flashvid_config (FlashVidConfig): The active config. Its ``method``
            attribute selects the policy.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Compressed tokens and their global
        indices, in the same format ``flashvid_compression`` returns.
    """
    method = _method_of(flashvid_config)
    fn = _resolve(_COMPRESSION_REGISTRY, method, "compression")
    return fn(
        video_features=video_features,
        cls_attention=cls_attention,
        flashvid_config=flashvid_config,
    )


def prune(flashvid_config: FlashVidConfig, **kwargs):
    """Dispatch inner-LLM pruning to the active method.

    Keyword arguments are forwarded verbatim, so a policy may accept the same
    signature as ``flashvid.utils.fastv_prune``.

    Args:
        flashvid_config (FlashVidConfig): The active config. Its ``method``
            attribute selects the policy.

    Returns:
        The tuple ``fastv_prune`` returns: ``(hidden_states, causal_mask,
        position_ids, cache_position, position_embeddings, keep_indices)``.
    """
    method = _method_of(flashvid_config)
    fn = _resolve(_LLM_PRUNING_REGISTRY, method, "inner-LLM pruning")
    return fn(flashvid_config=flashvid_config, **kwargs)


register_compression(DEFAULT_METHOD, flashvid_compression)
register_llm_pruning(DEFAULT_METHOD, fastv_prune)
