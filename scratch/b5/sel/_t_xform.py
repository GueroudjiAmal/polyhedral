"""Every transform's symbolic cost vs the oracle, cell by cell."""
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of
from oracle import oracle_cost

CASES = [
    ("band128",      [Iv(0, 128)]),
    ("band-mis",     [Iv(0, 32), Iv(100, 132)]),
    ("band-al",      [Iv(0, 32), Iv(128, 160)]),
    ("lattice-8",    [Ap(0, 8, 64)]),
    ("lattice-4",    [Ap(0, 4, 128)]),
    ("lattice-3",    [Ap(0, 3, 170)]),
    ("band+lat8",    [Iv(0, 64), Ap(0, 8, 64)]),
    ("c2-splitter",  [Iv(0, 24), Iv(500, 524), Ap(0, 2, 256)]),
    ("twodiag",      [Iv(0, 1), Iv(17, 18)]),
    ("causal",       [Iv(0, 512)]),
]
bad = n = skipped = 0
fails = {}
for name, pieces in CASES:
    D = DiagSpec(pieces)
    for N in (512, 1024):
        for BQ in (128, 64, 32, 16):
            for A in (128, 64, 32, 16):
                for cand in CANDIDATES:
                    got = cost_of(cand, D, N, BQ, A)
                    exp = oracle_cost(pieces, N, BQ, A, cand)
                    if got is None:
                        skipped += 1
                        continue
                    n += 1
                    if got != exp:
                        bad += 1
                        fails.setdefault((cand, name), []).append((N, BQ, A, got, exp))
print(f"symbolic vs oracle: {n-bad}/{n} exact, {skipped} declined (returned None)")
for (cand, nm), v in sorted(fails.items())[:12]:
    N, BQ, A, got, exp = v[0]
    print(f"  FAIL {cand:<18}{nm:<14} {len(v):>4} cells, e.g. N={N} {BQ}x{A}: {got} vs {exp}")
