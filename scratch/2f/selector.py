"""Predicate-driven transform selector.

Costs every candidate ANALYTICALLY from a symbolic displacement set, then takes
the argmin. The N x N mask is never built and no transformed mask is ever
tile-counted -- that constraint is what separates a selector from brute force.

Billing matches polyattn.transforms.tile_stats: axes zero-pad up to a multiple
of the tile; a live tile costs BQ*A except in a trailing partial column strip,
which is billed at its real width.
"""
from dset import DSet


def _ceil(a, b):
    return -(-a // b)


def _bill(live, live_last, ncols, BQ, A):
    pk = (-ncols) % A
    if pk:
        return (live - live_last) * BQ * A + live_last * BQ * (A - pk)
    return live * BQ * A


def _support(D, N):
    """(min, max) of the LIVE displacements, or None.

    A pair (q,kv) is live iff q-kv in D and both indices are in [0,N), so the
    live displacements are D cap [-(N-1), N-1] -- NOT D cap [0,N).  Searching
    only the non-negative half silently truncated every bidirectional mask; the
    bug was invisible because every mask validated against had D subset [0,inf).
    """
    lo = D.min_in(-(N - 1), N - 1)
    if lo is None:
        return None
    hi = lo
    b = D.bounds()
    for cand in range(min(b[1], N) - 1, lo - 1, -1):
        if D.hit(cand, cand):
            hi = cand
            break
    return lo, hi


# ---------------------------------------------------------------- identity --
def cost_identity(D, N, BQ, A):
    I, J = _ceil(N, BQ), _ceil(N, A)
    live = last = 0
    for i in range(I):
        qlo, qhi = i * BQ, min((i + 1) * BQ, N)
        for j in range(J):
            klo, khi = j * A, min((j + 1) * A, N)
            if D.hit(qlo - khi + 1, qhi - 1 - klo):
                live += 1
                if j == J - 1:
                    last += 1
    return _bill(live, last, N, BQ, A)


# ----------------------------------------------------------- residue perm ---
def _classes(N, s):
    """(cum, sizes) for the sort-by-(i mod s, i div s) permutation."""
    sizes = [(N - r + s - 1) // s for r in range(s)]
    cum, t = [], 0
    for z in sizes:
        cum.append(t)
        t += z
    return cum, sizes


def _segments(p0, p1, cum, sizes, s):
    """New-coord range [p0,p1] -> list of (residue r, t_lo, t_hi) in old coords."""
    out = []
    for r in range(s):
        a, b = cum[r], cum[r] + sizes[r] - 1
        lo, hi = max(p0, a), min(p1, b)
        if lo <= hi:
            out.append((r, lo - a, hi - a))
    return out


def cost_residue_perm(D, N, s, BQ, A):
    cum, sizes = _classes(N, s)
    I, J = _ceil(N, BQ), _ceil(N, A)
    qsegs = [_segments(i * BQ, min((i + 1) * BQ, N) - 1, cum, sizes, s) for i in range(I)]
    ksegs = [_segments(j * A, min((j + 1) * A, N) - 1, cum, sizes, s) for j in range(J)]
    live = last = 0
    for i in range(I):
        for j in range(J):
            hit = False
            for (r, t0, t1) in qsegs[i]:
                for (e, u0, u1) in ksegs[j]:
                    # q = t*s + r, kv = u*s + e  =>  q-kv = (r-e) + s*(t-u)
                    if D.hit_ap(r - e, s, t0 - u1, t1 - u0):
                        hit = True
                        break
                if hit:
                    break
            if hit:
                live += 1
                if j == J - 1:
                    last += 1
    return _bill(live, last, N, BQ, A)


# ------------------------------------------------------------- class B ------
def cost_fold(D, N, s, BQ, A, shear=False):
    sup = _support(D, N)
    if sup is None:
        return 0
    dlo, dhi = sup
    if shear:
        W = dhi - dlo + 1
        rng = lambda c0, c1: (dhi - c1, dhi - c0)      # col = dhi - d
    else:
        if dlo < 0:              # stridefold keeps only d >= 0; oracle declines
            return None
        for lo, hi in D.ivs:                            # every live d must be in sZ
            if hi - lo > 1 or lo % s:
                return None
        for r, st, lo, hi in D.aps:
            # The condition is that every element IN RANGE is divisible by s, not
            # that the progression is. Requiring st % s == 0 is stricter than the
            # maths whenever the range holds a single element: D = {8} written as
            # (r=8, st=5, lo=8, hi=9) is foldable by 4 and this test refused it.
            # Found by auditing against the oracle after b5 hit the same shape --
            # a decline that no ordinary mask exercises, so it read as coverage.
            first = max(lo, 0) + (r - max(lo, 0)) % st
            last_el = min(hi, N) - 1
            last_el -= (last_el - r) % st
            if first > last_el:
                continue                       # no elements in range: vacuous
            if first == last_el:
                if first % s:
                    return None                # the one element must be divisible
                continue
            if st % s or r % s:
                return None
        W = dhi // s + 1
        rng = lambda c0, c1: (c0 * s, c1 * s + s - 1)   # col = d // s
    I, J = _ceil(N, BQ), _ceil(W, A)
    live = last = 0
    for j in range(J):
        c0, c1 = j * A, min((j + 1) * A, W) - 1
        if c1 < c0:
            continue
        d_lo, d_hi = rng(c0, c1)
        for i in range(I):
            qlo, qhi = i * BQ, min((i + 1) * BQ, N)
            # (q,kv) live needs 0 <= q < N and 0 <= q-d < N, so for a row block
            # [qlo,qhi) the reachable displacements are [qlo-N+1, qhi-1].
            # Positive d clip the top rows, negative d clip the bottom ones --
            # the old code only handled the positive side.
            if D.hit(max(d_lo, qlo - N + 1), min(d_hi, qhi - 1)):
                live += 1
                if j == J - 1:
                    last += 1
    return _bill(live, last, W, BQ, A)


# ------------------------------------------------------------- selector -----
CANDIDATES = (["identity", "shear"]
              + [f"stridefold-{s}" for s in (2, 4, 8)]
              + [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)])


def costs(D, N, BQ, A):
    out = {"identity": cost_identity(D, N, BQ, A),
           "shear": cost_fold(D, N, 1, BQ, A, shear=True)}
    for s in (2, 4, 8):
        out[f"stridefold-{s}"] = cost_fold(D, N, s, BQ, A)
    for s in (2, 3, 4, 6, 8, 12, 16, 32):
        out[f"residue-perm-{s}"] = cost_residue_perm(D, N, s, BQ, A)
    return {k: v for k, v in out.items() if v is not None}


def select(D, N, BQ, A):
    c = costs(D, N, BQ, A)
    return min(c, key=lambda k: (c[k], CANDIDATES.index(k))), c
