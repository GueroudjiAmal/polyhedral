"""A transform selector derived from the predicate, not from the matrix.

Constraint that gives the exercise its point: never materialise the N x N mask,
never tile-count a materialised transformed mask. Everything below works from the
DIAGONAL OFFSET SET of the predicate, extracted in O(N) and thereafter treated as
a handful of integer runs.

Complexity: O(N) to extract and verify the offsets, then O(N/gcd(BQ,A)) per
candidate with no dependence on the number of live elements. See
`selector_scaling` for the measured curve.

The exact tile count for a diagonally-invariant mask is

    T(BQ, A) = sum over v in gZ with D n [v-A+1, v+BQ-1] != empty  of  n(v),
    n(v) = #{(i,j) : i*BQ - j*A = v, 0 <= i < N/BQ, 0 <= j < N/A},   g = gcd(BQ,A)

n(v) is the number of tile pairs realising diagonal offset v; it is NOT uniform,
which is the hole that sank an earlier version of this formula (docs/NOTES.md
sec 5e). It is computed exactly here rather than approximated.
"""
from math import gcd

import numpy as np

__all__ = ["select", "costs", "offsets_of", "tile_cost"]


# ------------------------------------------------------- predicate -> runs ----
def offsets_of(mask, N, checks=6):
    """Diagonal offset set as sorted inclusive runs, or None if not invariant.

    O(N * checks). Reads a handful of rows; never builds the square.
    """
    live = mask.row_cols(N - 1, N)
    d = np.flatnonzero(live[::-1])                 # d = (N-1) - kv
    if d.size == 0:
        return []
    # verify invariance on sampled rows rather than assuming it
    for q in np.linspace(N // 4, N - 2, checks).astype(int):
        want = np.zeros(N, dtype=bool)
        dd = d[d <= q]
        want[q - dd] = True
        if not np.array_equal(want, mask.row_cols(int(q), N)):
            return None
    brk = np.flatnonzero(np.diff(d) > 1)
    starts = np.concatenate(([d[0]], d[brk + 1]))
    ends = np.concatenate((d[brk], [d[-1]]))
    return list(zip(starts.tolist(), ends.tolist()))


def _merge(iv):
    if not iv:
        return []
    iv = sorted(iv)
    out = [list(iv[0])]
    for l, r in iv[1:]:
        if l <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], r)
        else:
            out.append([l, r])
    return [tuple(x) for x in out]


# ------------------------------------------------------------ exact cost -----
def _n_of_v(vs, N, BQ, A, g):
    """n(v), vectorised. The weight an earlier formula wrongly assumed uniform."""
    M = A // g
    lo = np.maximum(0, -(-vs // BQ))
    hi = np.minimum(N // BQ, (vs + N - 1) // BQ + 1)
    if M == 1:
        i0 = np.zeros_like(vs)
    else:
        inv = pow((BQ // g) % M, -1, M)
        i0 = ((vs // g) % M) * inv % M
    first = lo + ((i0 - lo) % M)
    cnt = np.where(first >= hi, 0, (hi - 1 - first) // M + 1)
    return np.maximum(cnt, 0)


def tile_cost(runs, N, BQ, A):
    """Exact elements computed for a diagonally-invariant mask, from its runs."""
    if not runs or N % BQ or N % A:
        return 0 if not runs else None
    g = gcd(BQ, A)
    iv = _merge([(l - BQ + 1, r + A - 1) for l, r in runs])
    total = 0
    for l, r in iv:
        lo = -((-l) // g) * g                       # first multiple of g >= l
        if lo > r:
            continue
        vs = np.arange(lo, r + 1, g, dtype=np.int64)
        total += int(_n_of_v(vs, N, BQ, A, g).sum())
    return total * BQ * A


def _shift_runs(runs, s, delta):
    """Runs of {u : u*s + delta in D}."""
    out = []
    for l, r in runs:
        a = -((-(l - delta)) // s)                  # ceil((l-delta)/s)
        b = (r - delta) // s
        if b >= a:
            out.append((a, b))
    return _merge(out)


def _cost_residue_perm(runs, N, s, BQ, A):
    """Exact, via block decomposition -- no permutation is ever materialised.

    Sorting both axes by (i mod s, i div s) turns the mask into an s x s grid of
    (N/s)-square blocks; block (c1,c2) is itself diagonally invariant with offset
    set {u : u*s + (c1-c2) in D}, and there are (s - |delta|) blocks per delta.
    Requires the tiles to nest inside the blocks, else the decomposition is not
    exact and the candidate is declared unavailable rather than approximated.
    """
    if N % s:
        return None
    nb = N // s
    if nb % BQ or nb % A:
        return None
    total = 0
    for delta in range(-(s - 1), s):
        sub = _shift_runs(runs, s, delta)
        if not sub:
            continue
        c = tile_cost(sub, nb, BQ, A)
        if c is None:
            return None
        total += (s - abs(delta)) * c
    return total


def _staircase(dvals, N, BQ, A, width, col_of):
    """Shared cost for class B, which produces an N x width staircase matrix.

    Column c holds diagonal offset `col_of(c)`; that column is live for rows
    q >= d. So a tile is non-empty iff some live column in it has d <= last row
    of the block. Trailing partial strip billed at TRUE width, per the shared
    convention in selector_oracle.
    """
    if width <= 0:
        return 0
    dmin = {}
    for d in dvals:
        c = col_of(d)
        blk = c // A
        if blk not in dmin or d < dmin[blk]:
            dmin[blk] = d
    nq = N // BQ
    total = 0
    for blk, d in dmin.items():
        i_min = max(0, -((-(d - BQ + 1)) // BQ))
        rows = max(0, nq - i_min)
        a_eff = min(A, width - blk * A)
        total += rows * BQ * a_eff
    return total


def _expand(runs, cap=1 << 22):
    n = sum(r - l + 1 for l, r in runs)
    if n > cap:
        return None
    return np.concatenate([np.arange(l, r + 1) for l, r in runs])


def _cost_shear(runs, N, BQ, A):
    """kv' = kv - q. transforms.t_shear indexes columns as maxD - d."""
    d = _expand(runs)
    if d is None:
        return None
    dmax = int(d[-1])
    return _staircase(d, N, BQ, A, dmax - int(d[0]) + 1, lambda x: dmax - x)


def _cost_stridefold(runs, N, s, BQ, A):
    """kv' = (q - kv)/s. Applicable only if every offset is a multiple of s."""
    d = _expand(runs)
    if d is None or d[0] < 0 or np.any(d % s):
        return None
    return _staircase(d, N, BQ, A, int(d[-1]) // s + 1, lambda x: x // s)


# --------------------------------------------------------------- selector ----
CANDIDATES = (["identity", "shear"]
              + [f"stridefold-{s}" for s in (2, 4, 8)]
              + [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)])


def costs(mask, N, BQ, A, runs=None):
    """{candidate: predicted elements} from the predicate alone."""
    runs = offsets_of(mask, N) if runs is None else runs
    if runs is None:
        return None                                  # not diagonally invariant
    out = {}
    for name in CANDIDATES:
        if name == "identity":
            c = tile_cost(runs, N, BQ, A)
        elif name == "shear":
            c = _cost_shear(runs, N, BQ, A)
        elif name.startswith("stridefold-"):
            c = _cost_stridefold(runs, N, int(name.split("-")[1]), BQ, A)
        else:
            c = _cost_residue_perm(runs, N, int(name.split("-")[-1]), BQ, A)
        if c is not None:
            out[name] = c
    return out


def select(mask, N, BQ, A):
    """The selector. Returns a candidate name.

    For a mask that is not diagonally invariant the closed form does not apply
    and this returns "identity" -- a stated limitation, not a guess dressed up.
    Experiments 5 and 6 found identity to be the oracle's own answer for every
    non-invariant mask in the zoo, so the fallback is not arbitrary, but it is
    still a fallback and is reported separately in the evaluation.
    """
    c = costs(mask, N, BQ, A)
    if not c:
        return "identity"
    return min(c, key=lambda k: (c[k], CANDIDATES.index(k)))
