"""d4 found regret 7.24 on short documents: identity is a bad answer there
because a short-document mask is nearly a narrow band and shear straightens it.
My cost engine is exact on those masks -- but exactness is not selection.
Does MY SELECTOR pick right?"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np, general as G
from oracle import tiles_cost, apply_xform

def ddoc(b, N, bidir=False):
    bb = np.array(list(b) + [N])
    d = np.searchsorted(bb, np.arange(N), side="right") - 1
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    same = d[:, None] == d[None, :]
    return same if bidir else (same & (kv <= q))

CASES = [(list(range(0, 1024, 4)), "docpack docs~4", False),
         (list(range(0, 1024, 8)), "docpack docs~8", False),
         (list(range(0, 1024, 8)), "bidoc docs~8", True),
         ([0, 2, 897], "docpack mixed 2/895", False),
         (list(range(0, 1024, 512)), "docpack docs~512 (control)", False)]
TIL = [128, 64, 32, 16]
print(f"{'mask':<28}{'cells':>6}{'agree':>8}{'meanReg':>9}{'maxReg':>9}  worst cell")
for b, nm, bi in CASES:
    N = 1024
    M0 = ddoc(b, N, bi)
    sp = G.BiDoc(b, nm) if bi else G.DocPack(b, nm)
    ok = tot = 0; regs = []; worst = (1.0, None)
    for BQ in TIL:
        for A in TIL:
            costs = {}
            for c in G.CANDIDATES:
                Mt = apply_xform(M0, c)
                if Mt is not None:
                    costs[c] = tiles_cost(Mt, BQ, A)
            pick, _ = G.select(sp, N, BQ, A)
            if pick is None or not costs:
                continue
            best = min(costs.values())
            tot += 1
            ok += (costs[pick] == best)
            r = costs[pick] / best
            regs.append(r)
            if r > worst[0]:
                bestc = min(costs, key=costs.get)
                worst = (r, f"{BQ}x{A} picked {pick}, best {bestc}")
    print(f"{nm:<28}{tot:>6}{ok/tot*100:>7.1f}%{sum(regs)/len(regs):>9.4f}"
          f"{max(regs):>9.4f}  {worst[1] or '-'}")
