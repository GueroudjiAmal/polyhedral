"""Re-run bidoc through the library BiDoc (with the AP-row walk), including
short documents -- the regime my first bidoc test could not reach."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
import general as G
from oracle import tiles_cost, apply_xform

def dense_bidoc(b, N):
    bb = np.array(list(b) + [N])
    d = np.searchsorted(bb, np.arange(N), side="right") - 1
    return d[:, None] == d[None, :]

CASES = [([0, 512, 1024, 1536], "bidoc-512"),
         ([0, 552, 1064, 1576], "bidoc-512+40 misaligned"),
         ([0, 300, 700, 1100, 1500], "bidoc-irregular"),
         (list(range(0, 1024, 8)), "bidoc-docs-of-8 (SHORT)"),
         ([0, 3, 5, 900], "bidoc-tiny-docs (SHORT)")]
TIL = [128, 64, 32, 16]
bad = n = 0
fails = {}
for b, nm in CASES:
    for N in (1024, 2048):
        b2 = [x for x in b if x < N] or [0]
        sp, M0 = G.BiDoc(b2, nm), dense_bidoc(b2, N)
        for c in G.CANDIDATES:
            Mt = apply_xform(M0, c)
            if Mt is None:
                continue
            for BQ in TIL:
                for A in TIL:
                    got = G.cost_of(c, sp, N, BQ, A)
                    if got is None:
                        continue
                    n += 1
                    exp = tiles_cost(Mt, BQ, A)
                    if got != exp:
                        bad += 1
                        fails.setdefault((c, nm), []).append((got, exp))
print(f"BiDoc incl. SHORT documents: {n-bad}/{n} exact")
for (c, nm), v in sorted(fails.items())[:8]:
    print(f"  FAIL {c:<18}{nm:<28}{len(v):>4} cells, e.g. {v[0][0]} vs {v[0][1]}")
