"""Unit tests for BudgetVID's budget arithmetic.

These cover the allocation math only, which is the part that decides whether a
comparison against a fixed-ratio baseline is honest. They deliberately need no
GPU, no torch and no model weights, so they can run anywhere:

    uv run python tests/test_allocation.py

`allocation.py` is loaded straight from its file so that importing it does not
drag in `budgetvid/__init__.py` and therefore torch. It only imports `math` and
`typing`, so this works and still exercises the real code.

The full-stack check is a different thing and needs a GPU: `allocation=uniform`
must reproduce `enable_flashvid=True` on the benchmarks. See budgetvid/README.md.
"""

import importlib.util
import math
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alloc = _load("budgetvid_allocation", REPO / "budgetvid" / "allocation.py")


@dataclass
class StubConfig:
    """The fields `allocation.py` reads, without pulling in torch."""

    retention_ratio: float = 0.25
    expansion: float = 1.25
    alpha: float = 0.7
    allocation: str = "uniform"
    enforce_budget: bool = True


def _check(condition, message):
    if not condition:
        raise AssertionError(message)


def test_uniform_matches_flashvid_formula():
    """`uniform` must reproduce flashvid/utils.py:46 for every parameter combo."""
    for num_visual_tokens in (49, 144, 196, 729):
        for retention_ratio in (0.10, 0.15, 0.20, 0.25, 0.333):
            for expansion in (1.0, 1.25):
                config = StubConfig(retention_ratio=retention_ratio, expansion=expansion)
                # This is the upstream expression, written out rather than imported,
                # so the test fails loudly if upstream ever changes it.
                expected = math.ceil(num_visual_tokens * retention_ratio * expansion)
                budgets = alloc.uniform_allocation(
                    segment_lengths=[3, 5, 2],
                    num_visual_tokens=num_visual_tokens,
                    config=config,
                )
                _check(budgets == [expected] * 3, f"uniform gave {budgets}, expected {[expected] * 3} " f"(nvt={num_visual_tokens}, r={retention_ratio}, e={expansion})")


def test_uniform_spends_exactly_the_global_budget():
    config = StubConfig()
    segment_lengths = [4, 9, 1, 18]
    num_frames = sum(segment_lengths)
    num_visual_tokens = 196

    budgets = alloc.uniform_allocation(segment_lengths=segment_lengths, num_visual_tokens=num_visual_tokens, config=config)
    used = alloc.spend(budgets, segment_lengths)
    allowed = alloc.global_budget(num_frames, num_visual_tokens, config)
    _check(used == allowed, f"uniform spent {used} of {allowed}")


def test_allocate_dispatches_and_validates():
    config = StubConfig()
    budgets = alloc.allocate(
        video_features=None,
        cls_attention=None,
        segment_lengths=[8, 8],
        num_visual_tokens=196,
        config=config,
    )
    _check(len(budgets) == 2, f"expected 2 budgets, got {budgets}")
    _check(all(isinstance(b, int) for b in budgets), f"budgets must be ints, got {budgets}")


def test_unknown_policy_is_rejected():
    config = StubConfig(allocation="does-not-exist")
    try:
        alloc.allocate(video_features=None, cls_attention=None, segment_lengths=[16], num_visual_tokens=196, config=config)
    except KeyError:
        return
    raise AssertionError("an unknown allocation policy should raise KeyError")


def test_overspend_is_rejected():
    """The guardrail that keeps comparisons honest."""
    config = StubConfig()
    segment_lengths = [8, 8]
    num_frames = 16
    num_visual_tokens = 196
    fair = alloc.per_frame_budget(num_visual_tokens, config)

    # One segment takes more than its share without another giving any back.
    budgets = [fair + 10, fair]
    try:
        alloc.validate(budgets, segment_lengths, num_frames, num_visual_tokens, config)
    except ValueError:
        pass
    else:
        raise AssertionError("overspending should raise while enforce_budget is set")

    # Opting out is allowed, but has to be deliberate.
    relaxed = StubConfig(enforce_budget=False)
    alloc.validate(budgets, segment_lengths, num_frames, num_visual_tokens, relaxed)


def test_reallocation_within_budget_is_accepted():
    """Taking from one segment to give to another is the whole point."""
    config = StubConfig()
    segment_lengths = [8, 8]
    num_frames = 16
    num_visual_tokens = 196
    fair = alloc.per_frame_budget(num_visual_tokens, config)

    budgets = [fair + 10, fair - 10]
    alloc.validate(budgets, segment_lengths, num_frames, num_visual_tokens, config)
    used = alloc.spend(budgets, segment_lengths)
    allowed = alloc.global_budget(num_frames, num_visual_tokens, config)
    _check(used == allowed, f"a fair swap should spend exactly the budget, got {used} of {allowed}")


def test_malformed_allocations_are_rejected():
    config = StubConfig()
    num_visual_tokens = 196

    cases = {
        "wrong length": ([10], [8, 8]),
        "zero budget": ([0, 10], [8, 8]),
        "negative budget": ([-1, 10], [8, 8]),
        "above frame capacity": ([num_visual_tokens + 1, 1], [1, 1]),
    }
    for label, (budgets, segment_lengths) in cases.items():
        try:
            alloc.validate(budgets, segment_lengths, sum(segment_lengths), num_visual_tokens, config)
        except ValueError:
            continue
        raise AssertionError(f"{label} should have been rejected: budgets={budgets}")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    print(f"\n{len(tests)} passed")
