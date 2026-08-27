"""Symbolic displacement sets.

A diagonally-invariant mask is exactly {(q,kv) : q-kv in D}. D is represented
NOT as a bitmap but as a union of integer intervals and arithmetic progressions,
so every query below is O(#components) regardless of N. This is the whole point:
the selector reasons about the predicate, never about an N x N array.

  interval (lo, hi)            -> {x : lo <= x < hi}
  ap       (r, s, lo, hi)      -> {x : x = r (mod s), lo <= x < hi}
"""
from math import gcd


class DSet:
    __slots__ = ("ivs", "aps")

    def __init__(self, ivs=(), aps=()):
        self.ivs = [(int(a), int(b)) for a, b in ivs if b > a]
        self.aps = [(int(r) % int(s), int(s), int(a), int(b))
                    for r, s, a, b in aps if b > a]

    def __or__(self, o):
        return DSet(self.ivs + o.ivs, self.aps + o.aps)

    # ---- O(#components) queries -------------------------------------------
    def hit(self, a, b):
        """Does D intersect the closed interval [a, b]?"""
        if b < a:
            return False
        for lo, hi in self.ivs:
            if max(a, lo) <= min(b, hi - 1):
                return True
        for r, s, lo, hi in self.aps:
            x, y = max(a, lo), min(b, hi - 1)
            if x > y:
                continue
            # smallest element >= x congruent to r mod s
            f = x + (r - x) % s
            if f <= y:
                return True
        return False

    def min_in(self, a, b):
        """Smallest element of D in [a, b], or None."""
        best = None
        for lo, hi in self.ivs:
            x, y = max(a, lo), min(b, hi - 1)
            if x <= y and (best is None or x < best):
                best = x
        for r, s, lo, hi in self.aps:
            x, y = max(a, lo), min(b, hi - 1)
            if x > y:
                continue
            f = x + (r - x) % s
            if f <= y and (best is None or f < best):
                best = f
        return best

    def hit_ap(self, base, step, k1, k2):
        """Does D contain any element base + step*k with k1 <= k <= k2?

        The tile of a residue-permuted mask has exactly this shape: its set of
        achievable displacements is an arithmetic progression, not an interval.
        """
        if k2 < k1:
            return False
        lo_v, hi_v = base + step * k1, base + step * k2
        for lo, hi in self.ivs:
            x, y = max(lo_v, lo), min(hi_v, hi - 1)
            if x > y:
                continue
            # any element of base+step*Z inside [x, y]?
            f = x + (base - x) % step
            if f <= y:
                return True
        for r, s, lo, hi in self.aps:
            x, y = max(lo_v, lo), min(hi_v, hi - 1)
            if x > y:
                continue
            # solve  n = base (mod step)  and  n = r (mod s)   -- CRT
            g = gcd(step, s)
            if (r - base) % g:
                continue
            l = step // g * s                      # lcm
            # find smallest n >= x with n = base (mod step), n = r (mod s)
            m = s // g
            t = 0 if m == 1 else ((r - base) // g) * pow(step // g, -1, m) % m
            n0 = base + step * t
            f = x + ((n0 - x) % l)          # smallest >= x with f = n0 (mod l)
            if f <= y:
                return True
        return False

    def bounds(self):
        vals = [(lo, hi) for lo, hi in self.ivs] + [(a, b) for _, _, a, b in self.aps]
        if not vals:
            return None
        return min(v[0] for v in vals), max(v[1] for v in vals)

    def count(self, N):
        """|D cap [0,N)| -- only used for reporting, O(#components)."""
        seen = set()
        for lo, hi in self.ivs:
            seen |= set(range(max(0, lo), min(N, hi)))
        for r, s, lo, hi in self.aps:
            a, b = max(0, lo), min(N, hi)
            if b > a:
                seen |= set(range(a + (r - a) % s, b, s))
        return len(seen)
