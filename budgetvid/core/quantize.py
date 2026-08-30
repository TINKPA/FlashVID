"""BudgetVID 2.0 -- measure quantization (CBA + MPQ).

Spec: notes/2026-08-28_method_budgetvid2_v1.html (frozen 2026-08-30).

Two stages sharing one farthest-point pass per frame:

* **CBA** (Curve-based Budget Allocation) -- one FPS pass yields, for free, the
  cost the quantizer will actually incur at *every* size (eq 2). Levelling the
  marginal cost across frames is reverse water-filling, and because each frame's
  curve is convex the exact argmin is a greedy heap (L2).
* **MPQ** (Mass-Preserving Quantization) -- each frame keeps its first ``b_t``
  seeds, every token joins its nearest seed, and the group is delivered as one
  token *plus its size*. The size becomes ``beta = log m``, an additive
  attention-score bias, which by L5 is an identity with reading the compressed
  set as the weighted measure it stands for.

Everything here is pure torch on plain tensors: no transformers, no CUDA
assumptions, no RNG. Determinism is load-bearing (a run must be bitwise
reproducible), so every argmax/argmin below breaks ties towards the smallest
index explicitly rather than trusting the backend's tie behaviour, which differs
between CPU and CUDA.
"""

from __future__ import annotations

import heapq

import torch


# --------------------------------------------------------------------------
# deterministic extrema
# --------------------------------------------------------------------------

def _argmax_first(v: torch.Tensor) -> torch.Tensor:
    """Row-wise argmax, ties to the SMALLEST index.

    ``torch.argmax`` does not promise which of several maxima it returns, and on
    a static frame the FPS residual is exactly 0 for many tokens at once, so
    ties are the normal case here rather than a corner case.
    """
    m = v.max(dim=-1, keepdim=True).values
    idx = torch.arange(v.shape[-1], device=v.device).expand_as(v)
    return torch.where(v >= m, idx, torch.full_like(idx, v.shape[-1])).min(dim=-1).values


def _argmin_first(v: torch.Tensor) -> torch.Tensor:
    """Row-wise argmin, ties to the smallest index. See ``_argmax_first``."""
    m = v.min(dim=-1, keepdim=True).values
    idx = torch.arange(v.shape[-1], device=v.device).expand_as(v)
    return torch.where(v <= m, idx, torch.full_like(idx, v.shape[-1])).min(dim=-1).values


# --------------------------------------------------------------------------
# Preliminary: the metric lift  (spec eq 1, B.2)
# --------------------------------------------------------------------------

def _rms(x: torch.Tensor, eps: float) -> torch.Tensor:
    """Root-mean-square over the feature axis, keepdim."""
    return x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()


def metric_lift(x: torch.Tensor, W_k: torch.Tensor | None, W_v: torch.Tensor | None,
                g: torch.Tensor | None, gamma_v: float = 1.0, eps: float = 1e-6):
    """Lift tokens into the geometry the decoder's first layer actually reads.

    ``Phi(x) = (W_k RN(x), sqrt(gamma_v) W_v RN(x))`` with
    ``RN(x) = (g * x) / rms(x)``. The RMSNorm belongs inside Phi because the key
    the decoder forms is ``W_k RN(x)`` and never ``W_k x``; the two differ by a
    per-token radial scale that a norm-free lift would read as content.

    Args:
        x: [..., d] tokens in the LLM's hidden space (post-projector).
        W_k: [d_k, d] or None. None on both W's means "no lift": distances are
            measured in the projector space itself, which is the metric ablation.
        W_v: [d_v, d] or None to drop the value half.
        g: [d] RMSNorm gain, or None to skip the normalization (the pre-freeze
            norm-free lift, kept as an ablation).
        gamma_v: weight of the value half against the key half.
        eps: RMSNorm epsilon.

    Returns:
        ``(K, s)`` -- lifted coordinates [..., d_k(+d_v)] in float32, and the
        per-token scale ``s = sqrt(d)/||x||`` [..., 1] used by the L1' centroid.
    """
    xf = x.float()
    s = 1.0 / _rms(xf, eps)                      # = sqrt(d)/||x||
    u = xf * s if g is not None else xf
    if g is not None:
        u = u * g.float()
    parts = []
    if W_k is not None:
        parts.append(u @ W_k.float().t())
    if W_v is not None and gamma_v > 0:
        parts.append((gamma_v ** 0.5) * (u @ W_v.float().t()))
    K = torch.cat(parts, dim=-1) if parts else u
    return K, s


# --------------------------------------------------------------------------
# Stage 1: CBA -- achievable-cost curves (spec eq 2, B.3)
# --------------------------------------------------------------------------

def fps_curves(K: torch.Tensor, b_max: int):
    """Farthest-point sampling with the cost curves it produces on the way.

    One pass over ``b_max`` steps, run for every frame at once. The seed sets are
    nested, so the same pass gives the frame's cost at *every* size:

        D_t(b) = sum_i min_{r<=b} d(x_i, seed_r)      (drives the allocation)
        r_t(b) = max_i min_{r<=b} d(x_i, seed_r)      (covering radius)

    ``D_t`` is the cost the quantizer of stage 2 will actually incur, not an
    estimate of it, because stage 2 reuses these very seeds.

    Args:
        K: [L, N_f, C] lifted coordinates.
        b_max: how far to run the curve; ``<= N_f``.

    Returns:
        ``(seeds, D, r)`` with seeds [L, b_max] (int64) and D, r [L, b_max]
        (float32), where column ``b-1`` holds the value at ``b`` seeds.
    """
    L, N_f, _ = K.shape
    b_max = int(min(b_max, N_f))
    if b_max < 1:
        raise ValueError(f"b_max must be >= 1, got {b_max}")

    seeds = torch.empty(L, b_max, dtype=torch.long, device=K.device)
    D = torch.empty(L, b_max, dtype=torch.float32, device=K.device)
    r = torch.empty(L, b_max, dtype=torch.float32, device=K.device)

    # Seed 1 is the token farthest from the frame's mean -- deterministic, and
    # the reason this needs no RNG at all.
    delta = (K - K.mean(1, keepdim=True)).norm(dim=-1)
    # Once a frame saturates -- every remaining token is an exact duplicate of
    # some seed -- the residual is all zeros and "farthest point" stops meaning
    # anything: the plain argmax then returns the SAME index at every further
    # step, and two seeds sharing an index means two output tokens claiming one
    # position. Water-filling never gets there (its saturation exit stops
    # spending on such a frame), but the even-split ablation has no such guard
    # and hit exactly this. Masking what is already taken keeps the seed
    # sequence a permutation prefix unconditionally, and changes nothing before
    # saturation, where the residuals are positive and the mask is inert.
    taken = torch.zeros(L, N_f, dtype=torch.bool, device=K.device)
    first = _argmax_first(delta)
    seeds[:, 0] = first
    taken.scatter_(1, first.unsqueeze(1), True)

    for b in range(b_max):
        cur = K.gather(1, seeds[:, b, None, None].expand(-1, 1, K.shape[-1]))
        dist = (K - cur).norm(dim=-1)
        delta = dist if b == 0 else torch.minimum(delta, dist)
        D[:, b] = delta.sum(1)
        r[:, b] = delta.max(1).values
        if b + 1 < b_max:
            nxt = _argmax_first(delta.masked_fill(taken, -1.0))
            seeds[:, b + 1] = nxt
            taken.scatter_(1, nxt.unsqueeze(1), True)
    return seeds, D, r


def lower_convex_envelope(D: torch.Tensor) -> torch.Tensor:
    """Lower convex envelope of each cost curve, over b = 1..b_max.

    The allocation is exactly optimal w.r.t. the envelope (L2) and only
    optimistic w.r.t. the raw curve; the gap is disclosed, not absorbed, so
    callers can report ``sum D_t(b_t) - sum D_bar_t(b_t)``.

    Args:
        D: [L, b_max] non-increasing cost curves.

    Returns:
        [L, b_max] envelope, float32, on the CPU-friendly same device.
    """
    Dc = D.detach().cpu().double()
    L, n = Dc.shape
    out = torch.empty_like(Dc)
    for t in range(L):
        y = Dc[t].tolist()
        hull = []                       # indices of the lower convex hull
        for i in range(n):
            while len(hull) >= 2:
                i0, i1 = hull[-2], hull[-1]
                # drop i1 if it sits on or above the chord (i0, i)
                if (y[i1] - y[i0]) * (i - i0) >= (y[i] - y[i0]) * (i1 - i0):
                    hull.pop()
                else:
                    break
            hull.append(i)
        env = list(y)
        for a, b in zip(hull, hull[1:]):
            for i in range(a, b + 1):
                w = (i - a) / (b - a) if b > a else 0.0
                env[i] = y[a] + w * (y[b] - y[a])
        out[t] = torch.tensor(env, dtype=torch.float64)
    return out.to(D.device, torch.float32)


# --------------------------------------------------------------------------
# Stage 1: CBA -- water-filling (spec eq 3, B.4)
# --------------------------------------------------------------------------

def waterfill(Dbar: torch.Tensor, B: int, r: torch.Tensor | None = None,
              caps: torch.Tensor | None = None) -> torch.Tensor:
    """Split B tokens across frames by levelling marginal cost.

    Every frame starts at the floor of 1 token (a frame may never vanish), then
    the remaining ``B - L`` tokens go one at a time to the frame with the largest
    marginal drop. Because each envelope is convex its gains are non-increasing,
    so this greedy IS the argmin (L2), not an approximation of it.

    A frame whose covering radius has reached 0 is saturated: every one of its
    tokens is already a seed's exact duplicate, so another token would buy
    nothing. It leaves the auction and its share is re-auctioned automatically.

    Args:
        Dbar: [L, b_max] convex envelopes of the cost curves.
        B: total token budget; must be >= L.
        r: [L, b_max] covering-radius curves, for the saturation exit. Optional.
        caps: [L] per-frame maximum (defaults to b_max).

    Returns:
        [L] int64 allocation. ``sum(b) == B`` unless every frame saturated
        first, in which case it is smaller and the caller should report it.
    """
    L, b_max = Dbar.shape
    if B < L:
        raise ValueError(f"B={B} < L={L}: a frame would get 0 tokens (spec floor)")
    D = Dbar.detach().cpu().double().tolist()
    rr = r.detach().cpu().double().tolist() if r is not None else None
    cap = [b_max] * L if caps is None else [int(c) for c in caps.tolist()]

    b = [1] * L
    heap: list[tuple[float, int]] = []

    def gain(t):
        """Marginal drop from b[t] to b[t]+1, or None if the frame is done."""
        if b[t] >= min(cap[t], b_max):
            return None
        if rr is not None and rr[t][b[t] - 1] <= 0.0:
            return None                      # saturated: nothing left to cover
        return D[t][b[t] - 1] - D[t][b[t]]

    for t in range(L):
        gt = gain(t)
        if gt is not None:
            heapq.heappush(heap, (-gt, t))   # max-heap via negation

    for _ in range(B - L):
        if not heap:
            break                            # every frame saturated; underspend
        _, t = heapq.heappop(heap)
        b[t] += 1
        gt = gain(t)
        if gt is not None:
            heapq.heappush(heap, (-gt, t))
    return torch.tensor(b, dtype=torch.long, device=Dbar.device)


# --------------------------------------------------------------------------
# Stage 2: MPQ -- quantize each frame (spec eq 4, B.5)
# --------------------------------------------------------------------------

def _assign(Kt: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Nearest-center assignment, ties to the smallest center index."""
    return _argmin_first(torch.cdist(Kt.unsqueeze(0), C.unsqueeze(0)).squeeze(0))


def lloyd_refine(Kt: torch.Tensor, sid: torch.Tensor, iters: int):
    """Move the centers to their groups' means, a few times (Lloyd / k-means).

    FPS is a k-CENTER heuristic: it takes the point farthest from everything
    chosen so far, which is an outlier by construction. That is the right move
    for the covering radius and the wrong one for the summed cost -- and the
    summed cost is what T1 bounds the attention error by, and what the
    allocation optimizes. The visible symptom is group imbalance: the outlier
    seeds hold a handful of tokens each while one seed absorbs the dense bulk
    of the frame, whose centroid is then the frame's mean wearing a mass of
    several hundred.

    Lloyd fixes exactly that: each center moves to its own group's mean, so
    centers migrate out of the sparse tail and into the mass. It monotonically
    decreases the k-means objective, which is the one the theory names.

    Args:
        Kt: [N_f, C] one frame's lifted coordinates.
        sid: [b] initial center indices (the FPS seeds).
        iters: how many Lloyd sweeps; 0 returns the FPS assignment unchanged.

    Returns:
        ``(assign, pos)`` -- the final assignment [N_f], and for each group the
        index of the token nearest its center (the medoid), which is what
        carries the delivered token's position identifier. A position has to be
        a real token's, never a coordinate average (v1.4's rule, inherited).
    """
    C = Kt[sid]
    a = _assign(Kt, C)
    for _ in range(max(0, iters)):
        b = C.shape[0]
        cnt = torch.bincount(a, minlength=b).clamp(min=1).unsqueeze(-1).float()
        new_C = torch.zeros_like(C)
        new_C.index_add_(0, a, Kt)
        new_C = new_C / cnt
        # An emptied center keeps its old position rather than collapsing to the
        # origin, which would swallow the whole frame on the next sweep.
        empty = torch.bincount(a, minlength=b) == 0
        if empty.any():
            new_C[empty] = C[empty]
        if torch.allclose(new_C, C):
            C = new_C
            break
        C = new_C
        a = _assign(Kt, C)
    # Each group's medoid must come from its OWN members. Taking the nearest
    # token to each center over ALL tokens lets two centers name the same token,
    # which delivers two output tokens at one position -- caught in production
    # by assemble()'s duplicate-index assertion after an hour of GPU, because
    # nothing else about the run looks wrong.
    b = C.shape[0]
    d = torch.cdist(Kt.unsqueeze(0), C.unsqueeze(0)).squeeze(0)          # [N, b]
    own = a.unsqueeze(1) != torch.arange(b, device=a.device)
    pos = _argmin_first(d.masked_fill(own, float("inf")).t())
    counts = torch.bincount(a, minlength=b)
    if bool((counts == 0).any()):
        # A center can end up with no members. It still owes one delivered
        # token, so give it the worst-covered token not already spoken for --
        # which is also the token that most wants a representative of its own.
        resid = d.gather(1, a.unsqueeze(1)).squeeze(1).clone()
        resid[pos[counts > 0]] = -1.0
        for j in torch.nonzero(counts == 0).flatten().tolist():
            t = int(_argmax_first(resid.unsqueeze(0)).item())
            pos[j] = t
            a[t] = j
            resid[t] = -1.0
    # Uniqueness is by construction above -- medoids are drawn from disjoint
    # groups, and the empty-group fill only takes tokens no medoid has claimed.
    # It is asserted anyway because the failure it guards against costs an hour
    # of GPU and surfaces far downstream, in assemble(), as a duplicate global
    # index. NOTE: the exact input that triggered the original failure was never
    # reproduced synthetically (bulk-plus-outlier and clumped clouds both stayed
    # clean under the old code), so this assert is the thing that would catch a
    # recurrence, not the test suite.
    if torch.unique(pos).numel() != pos.numel():
        raise AssertionError(
            f"lloyd_refine produced {pos.numel() - torch.unique(pos).numel()} duplicate "
            "medoid positions; groups are disjoint so this should be impossible")
    return a, pos


def quantize_frames(K: torch.Tensor, x: torch.Tensor, s: torch.Tensor,
                    seeds: torch.Tensor, b: torch.Tensor, centroid: str = "rms",
                    refine: int = 0):
    """Assign every token to its nearest seed and deliver one token per group.

    Args:
        K: [L, N_f, C] lifted coordinates (the decision space).
        x: [L, N_f, d] original tokens (the delivery space).
        s: [L, N_f, 1] RMS scales from ``metric_lift``.
        seeds: [L, b_max] FPS sequence.
        b: [L] per-frame number of seeds to keep.
        centroid: "rms" for the metric-space centroid direction at the group's
            mean norm (L1', the default), or "plain" for the unweighted mean in
            the original space (the ablation, and what every prior merge does).

    Returns:
        ``(feats, seed_idx, mass, cost)`` -- lists of length L holding, per
        frame, the delivered tokens [b_t, d], their seeds' within-frame indices
        [b_t], the group sizes [b_t] (summing to N_f), and the realized
        quantization cost ``D_t(b_t)`` as a float.
    """
    L, N_f, _ = K.shape
    feats, seed_idx, mass, cost = [], [], [], []
    for t in range(L):
        k = int(b[t])
        sid = seeds[t, :k]
        if refine:
            a, sid = lloyd_refine(K[t], sid, refine)
            a = a.clone()
        else:
            d = torch.cdist(K[t].unsqueeze(0), K[t, sid].unsqueeze(0)).squeeze(0)  # [N_f, k]
            a = _argmin_first(d)
            # Seed self-assignment, stated in the spec as the reason m_j >= 1
            # holds. Forcing it also removes the one way a group could come out
            # empty: two seeds with identical coordinates (duplicate tokens in a
            # static frame), where the tie-break would hand every token to the
            # first.
            a = a.clone()
        a[sid] = torch.arange(k, device=a.device)

        m = torch.bincount(a, minlength=k)
        num = torch.zeros(k, x.shape[-1], device=x.device, dtype=torch.float32)
        if centroid == "medoid":
            # Deliver the group's most central REAL token instead of an average
            # of its members. The measure being approximated is the same and the
            # mass channel is unchanged -- this is a weighted coreset of actual
            # points, which is the form the attention-coreset bounds are stated
            # for. It exists because a centroid is a vector the language model
            # has never seen: averaging tokens leaves the manifold the encoder
            # produces, and at a tight budget each average is over hundreds of
            # unrelated tokens. Whether that matters is measurable, and this is
            # the row that measures it.
            C = torch.zeros(k, K.shape[-1], device=K.device, dtype=torch.float32)
            C.index_add_(0, a, K[t])
            C = C / m.clamp(min=1).unsqueeze(-1).float()
            far = torch.cdist(K[t].unsqueeze(0), C.unsqueeze(0)).squeeze(0)
            # a token may only represent its own group
            far = far + (a.unsqueeze(1) != torch.arange(k, device=a.device)) * 1e9
            med = _argmin_first(far.t())
            feats.append(x[t, med].float())
            sid = med
        elif centroid == "rms":
            # Direction from the metric-space centroid (L1'), magnitude from the
            # group's mean norm. The direction is what the decoder reads: RMSNorm
            # erases whatever magnitude we deliver, so the scale is simply not
            # identifiable from the attention side. It is NOT free, though --
            # the residual stream carries the raw vector -- so it is pinned to
            # the group's mean token norm, which makes a group of one exactly
            # the identity and keeps a merged token the size of a token.
            num.index_add_(0, a, x[t].float() * s[t])
            nrm = torch.zeros(k, device=x.device, dtype=torch.float32)
            nrm.index_add_(0, a, x[t].float().norm(dim=-1))
            direction = num / num.norm(dim=-1, keepdim=True).clamp(min=1e-9)
            feats.append(direction * (nrm / m.clamp(min=1).float()).unsqueeze(-1))
        else:
            num.index_add_(0, a, x[t].float())
            feats.append(num / m.clamp(min=1).unsqueeze(-1).float())
        seed_idx.append(sid)
        mass.append(m)
        dd = torch.cdist(K[t].unsqueeze(0), K[t, sid].unsqueeze(0)).squeeze(0)
        cost.append(float(dd.gather(1, a.unsqueeze(1)).sum()))
    return feats, seed_idx, mass, cost


def compress_video(x: torch.Tensor, B: int, W_k=None, W_v=None, g=None,
                   gamma_v: float = 1.0, alloc: str = "waterfill",
                   centroid: str = "rms", b_max: int = 0, eps: float = 1e-6,
                   refine: int = 0):
    """Algorithm 2.0 end to end, on one video's visual tokens.

    Args:
        x: [L, N_f, d] visual tokens after the projector.
        B: token budget, ``L <= B < L*N_f``.
        W_k, W_v, g, gamma_v, eps: the metric lift, see ``metric_lift``.
        alloc: "waterfill" (CBA) or "even" (largest-remainder, the v0 split,
            kept as the allocation ablation).
        centroid: "rms" or "plain", see ``quantize_frames``.
        b_max: cap on the curve length; 0 means N_f (exact).
        refine: Lloyd sweeps after the FPS seeding, 0 = off (the frozen v1
            behaviour). See ``lloyd_refine`` for why this exists.

    Returns:
        dict with ``feats``/``seed_idx``/``mass`` (per frame), ``b`` [L],
        ``cost`` (realized), ``planned`` (envelope), ``radius`` [L].
    """
    L, N_f, _ = x.shape
    if not (L <= B <= L * N_f):
        raise ValueError(f"budget B={B} out of range for L={L}, N_f={N_f}")
    K, s = metric_lift(x, W_k, W_v, g, gamma_v, eps)
    bm = N_f if b_max <= 0 else int(min(b_max, N_f))
    seeds, D, r = fps_curves(K, bm)

    if alloc == "waterfill":
        Dbar = lower_convex_envelope(D)
        b = waterfill(Dbar, B, r=r)
        planned = float(sum(Dbar[t, int(b[t]) - 1] for t in range(L)))
    elif alloc == "even":
        from .budget import split_budget
        b = split_budget(B, L, N_f).to(x.device)
        if int(b.max()) > bm:
            raise ValueError(f"b_t={int(b.max())} exceeds curve length {bm}")
        planned = float(sum(D[t, int(b[t]) - 1] for t in range(L)))
    else:
        raise KeyError(f"unknown allocation '{alloc}'; known: waterfill, even")

    feats, seed_idx, mass, cost = quantize_frames(K, x, s, seeds, b, centroid, refine)
    return {"feats": feats, "seed_idx": seed_idx, "mass": mass, "b": b,
            "cost": float(sum(cost)), "planned": planned,
            "radius": torch.stack([r[t, int(b[t]) - 1] for t in range(L)]),
            "D": D, "r": r}


def curve_cost(D: torch.Tensor, b: torch.Tensor) -> float:
    """Cost of an allocation, read straight off the curves.

    ``D_t(b)`` is the cost the quantizer realizes at ``b`` seeds, so the cost of
    ANY allocation over the same seed sequence is a lookup -- including the one
    we did not take. That is what makes the water-filling-vs-even comparison
    (offline-replay note, P2) free rather than a second campaign: both numbers
    come out of the pass we already ran.
    """
    return float(sum(D[t, int(b[t]) - 1] for t in range(D.shape[0])))
