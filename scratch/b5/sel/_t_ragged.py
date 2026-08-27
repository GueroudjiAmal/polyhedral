"""Has my symbolic engine EVER been validated at N not divisible by the tile?
No -- every suite either used divisible N or skipped those cells outright
(`if N % BQ or N % A: continue`). d4's finding that the kernel asserts
divisibility made me look. This is the regime 2f's refinement is ABOUT."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of
import general as G
from oracle import oracle_cost, tiles_cost, apply_xform
import numpy as np

TIL = [128, 64, 32, 16]
bad = n = 0
fails = {}
print("A. diagonal engine at ragged N")
for pieces, nm in [([Iv(0, 128)], "band128"), ([Iv(0, 32), Iv(100, 132)], "twoband-mis"),
                   ([Ap(0, 8, 60)], "lattice-8"), ([Iv(0, 24), Ap(0, 2, 200)], "c2ish")]:
    for N in (500, 777, 1000, 1023, 1200):
        for BQ in TIL:
            for A in TIL:
                if N % BQ == 0 and N % A == 0:
                    continue                      # the regime already covered
                got = cost_of("identity", DiagSpec(pieces), N, BQ, A)
                exp = oracle_cost(pieces, N, BQ, A, "identity")
                if got is None:
                    continue
                n += 1
                if got != exp:
                    bad += 1
                    fails.setdefault(("diag", nm), []).append((N, BQ, A, got, exp))

print("B. general engine at ragged N")
def dsink(g, w, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & ((kv < g) | (q - kv < w))
for g_, w_ in [(4, 256), (16, 128)]:
    for N in (500, 777, 1000, 1023):
        M0 = dsink(g_, w_, N); sp = G.Sinks(g_, w_)
        for BQ in TIL:
            for A in TIL:
                if N % BQ == 0 and N % A == 0:
                    continue
                got = G.cost_of("identity", sp, N, BQ, A)
                if got is None:
                    continue
                n += 1
                exp = tiles_cost(M0, BQ, A)
                if got != exp:
                    bad += 1
                    fails.setdefault(("gen", f"sinks{g_}+w{w_}"), []).append((N, BQ, A, got, exp))

print(f"\nragged-N suite: {n-bad}/{n} exact")
for k, v in sorted(fails.items())[:8]:
    N, BQ, A, got, exp = v[0]
    print(f"  FAIL {k[0]}:{k[1]:<16}{len(v):>4} cells, e.g. N={N} {BQ}x{A}: {got} vs {exp}"
          f"  ({got/max(1,exp):.3f}x)")
