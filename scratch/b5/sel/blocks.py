"""Exact A-aligned block counting over a symbolic column set.

A column set is a union of PIECES, each either
    Iv(lo, hi)                  the integer interval [lo, hi)
    Ap(start, stride, count)    start, start+stride, ..., count terms
Kept symbolic so a lattice never has to be enumerated: that is the whole reason
this is sub-quadratic. Everything below is integer arithmetic; no arrays whose
length grows with N.
"""
from math import gcd


class Iv:
    __slots__ = ("lo", "hi")

    def __init__(self, lo, hi):
        self.lo, self.hi = lo, hi

    def empty(self):
        return self.hi <= self.lo

    def clip(self, n):
        return Iv(max(0, self.lo), min(n, self.hi))

    def __repr__(self):
        return f"Iv({self.lo},{self.hi})"


class Ap:
    __slots__ = ("start", "stride", "count")

    def __init__(self, start, stride, count):
        self.start, self.stride, self.count = start, stride, count

    def last(self):
        return self.start + (self.count - 1) * self.stride

    def empty(self):
        return self.count <= 0

    def clip(self, n):
        """Restrict to [0, n)."""
        if self.count <= 0:
            return Ap(0, 1, 0)
        s, t = self.start, self.stride
        m0 = 0 if s >= 0 else (-s + t - 1) // t
        last = self.last()
        m1 = self.count - 1 if last < n else (n - 1 - s) // t
        return Ap(s + m0 * t, t, max(0, m1 - m0 + 1))

    def __repr__(self):
        return f"Ap({self.start},{self.stride},{self.count})"


def _merge(ivs):
    """Sorted disjoint merge of [lo,hi) block intervals."""
    ivs = sorted((a, b) for a, b in ivs if b > a)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def count_blocks(pieces, A, ncols):
    """Distinct A-aligned blocks touched by the union of `pieces`. Exact.

    Complexity: O(#Iv pieces * log) for interval-only inputs -- the case the
    general engine uses. An Ap with stride <= A also costs O(1): consecutive
    terms cannot skip a block, so it IS a block interval. Only an Ap with
    stride > A costs more, its distinct blocks being enumerated (bounded by
    ncols/A). The diagonal engine never calls this, so that branch is
    diagnostics only and never on the selector's hot path.
    """
    ivs, loose = [], set()
    for p in pieces:
        q = p.clip(ncols)
        if q.empty():
            continue
        if isinstance(q, Iv):
            ivs.append((q.lo // A, (q.hi - 1) // A + 1))
        elif q.stride <= A:
            ivs.append((q.start // A, q.last() // A + 1))
        else:
            loose.update((q.start + m * q.stride) // A for m in range(q.count))
    merged = _merge(ivs)
    total = sum(b - a for a, b in merged)
    for b in loose:
        if not any(a <= b < c for a, c in merged):
            total += 1
    return total


def billed_cols(pieces, A, ncols):
    """Columns BILLED for the union of `pieces`: count_blocks * A, except that a
    trailing partial block is billed at its true width. Matches the fixed
    tile_stats convention (zero padding must not be charged for)."""
    n = count_blocks(pieces, A, ncols)
    if n == 0 or ncols % A == 0:
        return n * A
    last = (ncols - 1) // A
    touched = count_blocks(pieces, A, ncols) != count_blocks(pieces, A, last * A) \
        or any(_touches(p, last * A, ncols) for p in pieces)
    return n * A - ((A - ncols % A) if touched else 0)


def _touches(p, lo, hi):
    q = p.clip(hi)
    if q.empty():
        return False
    if isinstance(q, Iv):
        return q.hi > lo
    if q.last() < lo:
        return False
    m0 = max(0, (lo - q.start + q.stride - 1) // q.stride)
    return m0 < q.count and q.start + m0 * q.stride < hi
