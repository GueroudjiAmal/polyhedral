"""Agreement restricted to CONTESTED cells: those where the oracle's best and
second-best transform differ by more than a margin. On a cell where every
candidate costs the same, any selector agrees for free, so an overall agreement
number is dominated by cells that cannot distinguish anything. This is the
vacuity screen applied to selection rather than to symmetry.
"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, select
from oracle import oracle_cost
from evaluate import build_suite, TILES

MARGINS = (0.0, 0.01, 0.05, 0.20)
print("\n=== agreement on CONTESTED cells (oracle best vs 2nd-best gap > margin) ===")
print(f"{'margin':>8}{'cells':>8}{'agree':>8}{'meanReg':>9}{'maxReg':>8}")
rows = {m: [0, 0, [], 0.0] for m in MARGINS}
detail = []
for N in (1024, 2048):
    for name, pieces in build_suite(N):
        D = DiagSpec(pieces)
        for BQ in TILES:
            for A in TILES:
                costs = []
                for c in CANDIDATES:
                    oc = oracle_cost(pieces, N, BQ, A, c)
                    if oc is not None:
                        costs.append((oc, c))
                if len(costs) < 2:
                    continue
                costs.sort()
                best, second = costs[0][0], costs[1][0]
                if best == 0:
                    continue
                gap = second / best - 1.0
                pick, _ = select(D, N, BQ, A)
                if pick is None:
                    continue
                got = oracle_cost(pieces, N, BQ, A, pick)
                r = got / best
                ok = (got == best)
                for m in MARGINS:
                    if gap > m:
                        rows[m][0] += 1
                        rows[m][1] += ok
                        rows[m][2].append(r)
                if gap > 0.20 and not ok:
                    detail.append((name, N, BQ, A, pick, costs[0][1], r, gap))
for m in MARGINS:
    n, ag, regs, _ = rows[m]
    if n:
        print(f"{m*100:>7.0f}%{n:>8}{ag/n*100:>7.1f}%{sum(regs)/len(regs):>9.4f}"
              f"{max(regs):>8.4f}")
    else:
        print(f"{m*100:>7.0f}%{0:>8}   n/a")
print(f"\nwrong picks on cells with a >20% gap: {len(detail)}")
for d in detail[:10]:
    print(f"  {d[0]:<20} N={d[1]} {d[2]}x{d[3]}  picked {d[4]}, best {d[5]},"
          f" regret {d[6]:.3f}, gap {d[7]*100:.0f}%")
