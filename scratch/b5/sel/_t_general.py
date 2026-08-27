"""Validate the general (non-diagonally-invariant) engine against brute force."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
import general as G
from oracle import tiles_cost, apply_xform

def dense_sinks(g, w, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & ((kv < g) | (q - kv < w))

def dense_docpack(bounds, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    b = np.array(bounds + [N])
    doc = np.searchsorted(b, np.arange(N), side="right") - 1
    return (kv <= q) & (doc[:, None] == doc[None, :])

CASES = []
for g_, w_ in [(4, 256), (16, 128), (4, 1024)]:
    CASES.append((f"sinks{g_}+win{w_}", G.Sinks(g_, w_), lambda N, g_=g_, w_=w_: dense_sinks(g_, w_, N)))
for bl, nm in [([0, 300, 700, 1100, 1500], "docpack-irregular"),
               ([0, 512, 1024, 1536], "docpack-512")]:
    CASES.append((nm, G.DocPack(bl, nm), lambda N, bl=bl: dense_docpack(bl, N)))

TILES = [128, 64, 32, 16]
bad = n = 0
fails = {}
for name, spec, mk in CASES:
    for N in (1024, 2048):
        M0 = mk(N)
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
                        fails.setdefault((cand, name), []).append((N, BQ, A, got, exp))
print(f"general engine vs oracle: {n-bad}/{n} exact")
for (c, nm), v in sorted(fails.items())[:10]:
    N, BQ, A, got, exp = v[0]
    print(f"  FAIL {c:<16}{nm:<20}{len(v):>4} cells, e.g. N={N} {BQ}x{A}: {got} vs {exp}")
