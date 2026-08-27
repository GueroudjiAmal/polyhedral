"""Verify 2f's load-imbalance counterexample to MY corollary, by brute force.

My corollary claimed every transform-dependent hardware term is MEMORY. 2f says
max-tiles-per-row-block (makespan) is transform-dependent, compute-side, and not
a function of total tiles. Checking directly on materialised masks.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def per_prog_tiles(M, BQ, A):
    """Tiles per row-block = per program. Returns (total, max)."""
    nq, nk = M.shape
    pq, pk = (-nq) % BQ, (-nk) % A
    P = np.pad(M, ((0, pq), (0, pk)))
    t = P.reshape(-1, BQ, P.shape[1] // A, A).any(axis=(1, 3))
    per = t.sum(axis=1)
    return int(per.sum()), int(per.max())

CANDS = ["identity", "residue-perm-2", "residue-perm-4", "residue-perm-8"]
MASKS = {"local256+str8": lambda N: [Iv(0, 256), Ap(0, 8, N // 8)],
         "local128+str4":  lambda N: [Iv(0, 128), Ap(0, 4, N // 4)],
         "dilated-8":      lambda N: [Ap(0, 8, N // 8)],
         "window-128":     lambda N: [Iv(0, 128)]}
TIL = [(128,128),(128,32),(128,16),(64,64),(32,32),(16,16)]

dis = tot = 0
print(f"{'mask':<16}{'N':>6}{'tile':>9}{'count argmin':>16}{'makespan argmin':>18}{'':>4}")
for N in (1024, 2048):
    for nm, mk in MASKS.items():
        M0 = dense(mk(N), N)
        for BQ, A in TIL:
            stats = {}
            for c in CANDS:
                Mt = apply_xform(M0, c)
                if Mt is not None:
                    stats[c] = per_prog_tiles(Mt, BQ, A)
            if len(stats) < 2:
                continue
            tot += 1
            by_tot = min(stats, key=lambda c: stats[c][0])
            by_max = min(stats, key=lambda c: stats[c][1])
            if by_tot != by_max:
                dis += 1
                print(f"{nm:<16}{N:>6}{f'{BQ}x{A}':>9}"
                      f"{by_tot + f' (tot {stats[by_tot][0]}, max {stats[by_tot][1]})':>16}"
                      f"{by_max + f' (tot {stats[by_max][0]}, max {stats[by_max][1]})':>18}"
                      .replace("residue-perm-", "rp"))
print(f"\ndisagreements: {dis}/{tot} cells")
print("\nWAVE CHECK -- 2f's own caveat: makespan is the right model only when")
print("programs are comparable to SM count (108 on A100).")
for N in (2048, 4096):
    for BQ in (128, 32, 16):
        nprog = (N // BQ) * 8
        print(f"  N={N} BQ={BQ:>3} BH=8 -> {nprog:>5} programs = {nprog/108:>5.1f} waves"
              f"   {'makespan dominates' if nprog/108 < 4 else 'averages out'}")
