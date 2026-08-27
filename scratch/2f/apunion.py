"""Independent check of b5's retraction.

Claim: diagonal invariance is NOT required. It suffices that the mask can supply,
in closed form, the union of live kv columns over an ARITHMETIC PROGRESSION of
query rows -- because under residue-perm-s a permuted row-block is exactly an AP
of original rows, and each original column interval maps to <= s intervals.

Tested here on sinks(g,w), which is NOT diagonally invariant (the prefix sits at
absolute columns), against a materialised oracle.
"""
import numpy as np


def sinks_union_ap(g, w, N, r, s, t0, t1):
    """Union of live kv columns over rows {t*s + r : t0 <= t <= t1}, as intervals."""
    qs = [t * s + r for t in range(t0, t1 + 1) if 0 <= t * s + r < N]
    if not qs:
        return []
    out = [(0, min(g, max(qs) + 1))]
    if s <= w:                       # windows overlap: one interval
        out.append((max(0, min(qs) - w + 1), max(qs) + 1))
    else:                            # disjoint windows, one per row
        for q in qs:
            out.append((max(0, q - w + 1), q + 1))
    return [(a, b) for a, b in out if b > a]


def perm_image(a, b, s, cum):
    """Image of the column interval [a,b) under i -> cum[i%s] + i//s."""
    out = []
    for c in range(s):
        lo = a + (c - a) % s                     # first index >= a with i%s==c
        if lo >= b:
            continue
        hi = b - 1 - (b - 1 - c) % s             # last index < b with i%s==c
        out.append((cum[c] + lo // s, cum[c] + hi // s + 1))
    return out


def cost_sinks_residue_perm(g, w, N, s, BQ, A):
    """Analytic cost, no N x N array anywhere."""
    sizes = [(N - r + s - 1) // s for r in range(s)]
    cum, t = [], 0
    for z in sizes:
        cum.append(t); t += z
    J = -(-N // A)
    live = last = 0
    for i in range(-(-N // BQ)):
        p0, p1 = i * BQ, min((i + 1) * BQ, N) - 1
        blocks = set()
        for r in range(s):                        # row-block -> APs of original rows
            a, b = cum[r], cum[r] + sizes[r] - 1
            lo, hi = max(p0, a), min(p1, b)
            if lo > hi:
                continue
            for (ca, cb) in sinks_union_ap(g, w, N, r, s, lo - a, hi - a):
                for (pa, pb) in perm_image(ca, cb, s, cum):
                    blocks.update(range(pa // A, (pb - 1) // A + 1))
        live += len(blocks)
        last += (J - 1) in blocks
    pk = (-N) % A
    return (live - last) * BQ * A + last * BQ * (A - pk) if pk else live * BQ * A
