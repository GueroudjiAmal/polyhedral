"""d4's sharp test: does the AP-union primitive handle BIDOC exactly?
Bidoc is bidirectional packed documents -- M = M^T, NOT causal, NOT diagonally
invariant. It is the mask that broke the max-law while staying perfectly
symmetric, so it is the case most likely to expose a hidden causal assumption.

Per the rule we just promoted: my earlier suites all shared 'causal' and all
shared 'boundaries at multiples of 512'. Both are varied here deliberately.
"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
from blocks import Iv
import general as G
from oracle import tiles_cost, apply_xform


class BiDoc(G.RowSpec):
    """Bidirectional within a document: live iff q and kv share a document."""

    def __init__(self, bounds, name="bidoc"):
        self.b = list(bounds)
        self.name = name

    def _doc(self, q):
        lo, hi = 0, None
        for x in self.b:
            if x <= q:
                lo = x
            else:
                hi = x
                break
        return lo, hi

    def cols(self, q0, q1, step, N):
        out, q = [], q0
        qmax = q0 + ((q1 - 1 - q0) // step) * step
        while q <= qmax:
            lo, hi = self._doc(q)
            out.append(Iv(lo, hi if hi is not None else N))
            q = (hi if hi is not None and hi <= qmax else qmax + 1)
        return out


def dense_bidoc(bounds, N):
    b = np.array(list(bounds) + [N])
    doc = np.searchsorted(b, np.arange(N), side="right") - 1
    return doc[:, None] == doc[None, :]


CASES = [
    ([0, 512, 1024, 1536], "bidoc-512 (aligned)"),
    ([0, 552, 1064, 1576], "bidoc-512+40 (misaligned)"),
    ([0, 300, 700, 1100, 1500, 1900], "bidoc-irregular"),
    ([0, 7, 900, 1001], "bidoc-tiny-first-doc"),
]
TILES = [128, 64, 32, 16]
bad = n = 0
fails = {}
for bl, nm in CASES:
    for N in (1024, 2048):
        bl2 = [x for x in bl if x < N] or [0]
        spec, M0 = BiDoc(bl2, nm), dense_bidoc(bl2, N)
        for cand in G.CANDIDATES:
            Mt = apply_xform(M0, cand)
            if Mt is None:
                continue
            for BQ in TILES:
                for A in TILES:
                    got = G.cost_of(cand, spec, N, BQ, A)
                    if got is None:
                        continue
                    exp = tiles_cost(Mt, BQ, A)
                    n += 1
                    if got != exp:
                        bad += 1
                        fails.setdefault((cand, nm), []).append((N, BQ, A, got, exp))
print(f"BIDOC via AP-union primitive: {n-bad}/{n} exact")
for (c, nm), v in sorted(fails.items())[:8]:
    N, BQ, A, got, exp = v[0]
    print(f"  FAIL {c:<16}{nm:<26}{len(v):>4} cells, e.g. N={N} {BQ}x{A}: {got} vs {exp}")

# symmetry check: bidoc is the mask that broke the max-law while staying symmetric
print("\nsanity: cost symmetry on bidoc under identity (should be 0 -- M = M^T)")
for bl, nm in CASES[:2]:
    N = 2048
    bl2 = [x for x in bl if x < N] or [0]
    spec = BiDoc(bl2, nm)
    g = {(b, a): G.cost_of("identity", spec, N, b, a) for b in TILES for a in TILES}
    sym = max(abs(g[(b, a)] - g[(a, b)]) for b in TILES for a in TILES)
    spread = max(max(v) - min(v) for v in
                 ([g[(b, a)] for b in TILES for a in TILES if max(b, a) == m] for m in TILES))
    print(f"  {nm:<28} symmetry {sym}   max-class spread {spread}"
          f"  ({'max-law holds' if spread == 0 else 'max-law VIOLATED, as expected'})")
