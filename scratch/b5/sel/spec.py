"""Symbolic mask specs and exact tiling cost, without materialising anything.

DIAGONAL masks -- live(q,kv) <=> q-kv in D -- are the family where transforms
matter, and they get the analytic engine:

  cost(BQ,A) = BQ*A * sum over v in g*Z with D cap [v-BQ+1, v+A-1] nonempty of n(v)
  n(v) = #{x : x = 0 mod BQ, x = -v mod A, max(0,-v) <= x < min(N,N-v)},  g=gcd(BQ,A)

n(v) is O(1) by CRT, and v ranges over O(span(D)/g) values, so this is O(N/g) at
worst and O(1) for bounded D. It never touches an N x N array.

NON-DIAGONAL masks (sinks, docpack, bidoc) have interval column sets and get the
per-row-block engine: O(N/BQ * pieces).
"""
from math import gcd
from blocks import Iv, Ap, count_blocks


def _egcd(a, b):
    if b == 0:
        return a, 1, 0
    g, x, y = _egcd(b, a % b)
    return g, y, x - (a // b) * y


def n_of_v(v, N, BQ, A):
    """How many (row-block, col-block) pairs realise diagonal offset v. O(1)."""
    g = gcd(BQ, A)
    if v % g:
        return 0
    lo, hi = max(0, -v), min(N, N - v)
    if hi <= lo:
        return 0
    L = BQ // g * A
    # x = BQ*k with BQ*k = -v (mod A)  ->  k = (-v/g)*inv(BQ/g)  (mod A/g)
    Ag, BQg = A // g, BQ // g
    _, inv, _ = _egcd(BQg % Ag, Ag) if Ag > 1 else (1, 0, 0)
    k0 = 0 if Ag == 1 else ((-v // g) * inv) % Ag
    x0 = BQ * k0
    if x0 < lo:
        x0 += ((lo - x0) + L - 1) // L * L
    return 0 if x0 >= hi else (hi - 1 - x0) // L + 1


class DiagSpec:
    """live(q,kv) <=> (q-kv) in D, on [0,N)^2. D is a union of Iv/Ap pieces."""

    def __init__(self, pieces, name="diag"):
        self.pieces, self.name = list(pieces), name

    # --- predicate queries, all O(#pieces) ---------------------------------
    def hits(self, a, b):
        """Does D meet the integer interval [a, b]?"""
        for p in self.pieces:
            if isinstance(p, Iv):
                if p.lo <= b and a < p.hi:
                    return True
            else:
                if p.count <= 0:
                    continue
                lo, hi = max(a, p.start), min(b, p.last())
                if hi >= lo:
                    m0 = (lo - p.start + p.stride - 1) // p.stride
                    if p.start + m0 * p.stride <= hi:
                        return True
        return False

    def span(self):
        los, his = [], []
        for p in self.pieces:
            if isinstance(p, Iv):
                if not p.empty():
                    los.append(p.lo); his.append(p.hi - 1)
            elif p.count > 0:
                los.append(p.start); his.append(p.last())
        return (min(los), max(his)) if los else (0, -1)

    def max_le(self, x):
        """Largest element of D that is <= x, or None."""
        best = None
        for p in self.pieces:
            if isinstance(p, Iv):
                if not p.empty() and p.lo <= x:
                    c = min(p.hi - 1, x)
                    best = c if best is None else max(best, c)
            elif p.count > 0 and p.start <= x:
                m = min(p.count - 1, (x - p.start) // p.stride)
                c = p.start + m * p.stride
                best = c if best is None else max(best, c)
        return best

    # --- cost -------------------------------------------------------------
    def cost(self, N, BQ, A):
        """Exact ONLY when BQ | N and A | N.

        The n(v) reorganisation bills BQ*A for every live tile. At ragged N the
        final row-block and column-block are partial, and the oracle bills the
        trailing strip at its true width, so this over-counts by ~1%. It used to
        return that wrong number SILENTLY; it now refuses. Ragged N is exactly
        the regime 2f's padded-extent refinement is about, and it is also the
        regime the Triton kernel asserts away -- so it is modelled by nobody and
        implemented by nobody. Named in the limitations rather than papered over.
        """
        if N % BQ or N % A:
            raise NotImplementedError(
                f"ragged N unsupported: N={N} not divisible by BQ={BQ} and A={A}. "
                "Cost would be over-counted ~1% by billing partial edge tiles in full.")
        g = gcd(BQ, A)
        lo, hi = self.span()
        if hi < lo:
            return 0
        v0 = ((lo - (A - 1)) // g) * g
        v1 = hi + BQ - 1
        tot = 0
        v = v0
        while v <= v1:
            if self.hits(v - BQ + 1, v + A - 1):
                tot += n_of_v(v, N, BQ, A)
            v += g
        return BQ * A * tot

    def live(self, N):
        """Number of live (q,kv) pairs on [0,N)^2: sum of N-|d| over the UNION
        of offsets, clipped to (-N, N).

        Three bugs lived here, each invisible to the test before it:
          v1 summed pieces independently -> double-counted Iv/Ap overlap (8.9%
             on local256+str8). Found by cross-checking a published waste table.
          v2 deduped Ap elements only against merged INTERVALS, not against other
             Aps -> Ap(0,5,50) with Ap(0,7,50) double-counted their multiples of 35.
          v2 also clipped every piece to [0,N), silently DISCARDING negative
             offsets, so any non-causal mask lost half its elements.
        Both v2 bugs were found only by testing live() DIRECTLY against brute
        force. No regime probe would have found them: the quantity had no test.

        Offsets are enumerated (at most 2N of them), which is O(N) and off the
        hot path -- select() ranks costs and never calls this.
        """
        offs = set()
        for p in self.pieces:
            if isinstance(p, Iv):
                lo, hi = max(-(N - 1), p.lo), min(N, p.hi)
                offs.update(range(lo, hi))
            elif p.count > 0:
                for m in range(p.count):
                    d = p.start + m * p.stride
                    if -(N - 1) <= d < N:
                        offs.add(d)
        return sum(N - abs(d) for d in offs)
