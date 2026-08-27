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

    def residue_runs(self, s):
        """{rho: [Ap(start, s, count)]} -- D split by residue class mod s.
        Each Iv contributes one run per residue; each Ap contributes runs only
        for the residues its own stride reaches."""
        out = {}
        for p in self.pieces:
            if isinstance(p, Iv):
                if p.empty():
                    continue
                for rho in range(s):
                    first = p.lo + ((rho - p.lo) % s)
                    if first < p.hi:
                        cnt = (p.hi - 1 - first) // s + 1
                        out.setdefault(rho, []).append(Ap(first, s, cnt))
            else:
                if p.count <= 0:
                    continue
                step = p.stride
                d = gcd(step, s)
                # terms cycle through residues rho = (start + m*step) mod s with period s/d
                per = s // d
                for m in range(min(per, p.count)):
                    rho = (p.start + m * step) % s
                    cnt = (p.count - 1 - m) // per + 1
                    out.setdefault(rho, []).append(
                        Ap(p.start + m * step, per * step, cnt))
        return out

    # --- cost -------------------------------------------------------------
    def cost(self, N, BQ, A):
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
        tot = 0
        for p in self.pieces:
            q = p.clip(N)
            if isinstance(q, Iv):
                for d in range(q.lo, q.hi):
                    tot += N - d
            else:
                for m in range(q.count):
                    tot += N - (q.start + m * q.stride)
        return tot
