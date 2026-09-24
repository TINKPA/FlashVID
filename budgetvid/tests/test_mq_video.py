"""alloc="video" (video-level coreset) tests. No GPU, no model:

    uv run --no-project --python 3.11 --with torch python budgetvid/tests/test_mq_video.py

Design: notes/2026-09-24_note_video-level-coreset-design.html
"""

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from budgetvid.core.assembly import assemble                          # noqa: E402
from budgetvid.core.quantize import compress_video, metric_lift, video_fps  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail and not cond else ""))


def main():
    torch.manual_seed(0)
    L, N_f, d = 8, 24, 16
    x = torch.randn(L, N_f, d)

    for centroid in ("rms", "medoid", "plain"):
        for B in (L, L + 1, 40, L * N_f // 2, L * N_f - 3):
            out = compress_video(x, B, alloc="video", centroid=centroid)
            b = out["b"]
            g = torch.cat([t * N_f + out["seed_idx"][t] for t in range(L)])
            tag = f"{centroid} B={B}"
            check(f"budget spent exactly ({tag})", int(b.sum()) == B, f"{int(b.sum())}")
            check(f"one token per frame at least ({tag})", bool((b >= 1).all()), str(b.tolist()))
            check(f"no duplicate positions ({tag})", torch.unique(g).numel() == g.numel())
            check(f"mass sums to L*N_f ({tag})",
                  int(sum(m.sum() for m in out["mass"])) == L * N_f)
            check(f"indices in range ({tag})",
                  all(int(i.min()) >= 0 and int(i.max()) < N_f for i in out["seed_idx"] if i.numel()))
            tok, gi, ms = assemble(x, [torch.empty(0, dtype=torch.long)] * L,
                                   merged=list(zip(out["feats"], out["seed_idx"])),
                                   expected_total=B, masses=out["mass"])
            check(f"assembles ({tag})", tok.shape[0] == B and bool((gi[1:] > gi[:-1]).all()))

    # determinism
    a = compress_video(x, 50, alloc="video", centroid="medoid")
    b2 = compress_video(x, 50, alloc="video", centroid="medoid")
    check("deterministic", all(torch.equal(p, q) for p, q in zip(a["seed_idx"], b2["seed_idx"])))

    # L=1: the video IS the frame, so the seeds are the per-frame FPS prefix
    x1 = torch.randn(1, N_f, d)
    for B in (1, 5, 12):
        v = compress_video(x1, B, alloc="video", centroid="rms")
        e = compress_video(x1, B, alloc="even", centroid="rms")
        check(f"L=1 equals per-frame (B={B})",
              torch.equal(v["seed_idx"][0].sort().values, e["seed_idx"][0].sort().values)
              and torch.allclose(torch.cat(v["feats"]).sum(0), torch.cat(e["feats"]).sum(0), atol=1e-4))

    # repeated content: a video whose frames are exact copies needs only the floor
    # to cover it, so every extra seed lands on new content, never on a copy.
    # Static frames (every token identical, so the floor covers them completely);
    # the last frame gains 4 distinct new tokens. Its floor seed takes one of them,
    # and the 3 extra seeds must go to the other three -- not to any copy.
    xs = torch.ones(L, N_f, d)
    xs[-1, :4] = 10.0 * torch.randn(4, d)
    K, _ = metric_lift(xs, None, None, None)
    seeds, D, r = video_fps(K, L + 3)
    new = set(range((L - 1) * N_f, (L - 1) * N_f + 4))
    check("new content gets the extra seeds",
          set(seeds[L - 1:].tolist()) == new, str(seeds.tolist()))
    check("static frames are fully covered by the floor + new seeds", float(r[-1]) == 0.0)
    check("cost curve non-increasing", bool((D[1:] <= D[:-1] + 1e-4).all()))

    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
