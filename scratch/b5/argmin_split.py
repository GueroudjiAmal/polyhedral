"""Search for a diagonally-invariant mask whose ARGMIN transform differs between
two tile shapes with the SAME max(BQ,A).  That would break C2, not just C3."""
import numpy as np, itertools
from polyattn import transforms
from polyattn.explore import Custom
TILES = [128, 64, 32, 16]
PAIRS = [(bq, a) for bq in TILES for a in TILES]

def probe(off, w, s, N=1024):
    m = Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < w) | (((q-kv) >= off) & ((q-kv) < off+w)) | ((q-kv) % s == 0)), "x")
    M = m.dense(N); live = int(M.sum())
    variants = [(n, fn(M)[0]) for n, fn in transforms.candidates()
                if fn(M)[1] and fn(M)[1][0] == "A"]
    cells = {}
    for bq, a in PAIRS:
        cells[(bq, a)] = min((transforms.tile_stats(Mt, bq, a)[1]/live, n)
                             for n, Mt in variants)
    hits = []
    for mx in TILES:
        ns = {cells[p][1] for p in PAIRS if max(p) == mx}
        if len(ns) > 1:
            hits.append((mx, sorted(ns), {p: cells[p] for p in PAIRS if max(p) == mx}))
    return hits, cells

found = 0
for off, w, s in itertools.product([500, 700, 900, 1000, 1004, 1010, 1100],
                                   [24, 40, 100, 200, 252, 300],
                                   [2, 3, 4, 6, 8]):
    hits, cells = probe(off, w, s)
    if hits:
        found += 1
        print(f"\nSPLIT: band0-{w} + band{off}-{off+w} + stride{s}")
        for mx, ns, cs in hits:
            print(f"  max={mx}: argmins {ns}")
            for p in sorted(cs, key=lambda p: -p[0]):
                print(f"    {p[0]:>4}x{p[1]:<4} -> {cs[p][1]:<16}{cs[p][0]:.4f}")
        if found >= 3:
            break
print(f"\nsplits found: {found}")
