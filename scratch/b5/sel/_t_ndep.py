"""Confirm the N-dependent argmin flip against BRUTE FORCE, not my engine.
NOTES 5c's table is at N=2048 only. If the argmin moves with N, selection
depends on (predicate, BQ, A, N), not (predicate, BQ, A)."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from oracle import oracle_cost
CANDS = ["identity", "residue-perm-2", "residue-perm-4", "residue-perm-8"]
print("BRUTE FORCE, local256+str8, class-A candidates, cost (elements)")
for BQ, A in [(128, 128), (16, 16)]:
    print(f"\n  tile {BQ}x{A}")
    print(f"{'N':>7}" + "".join(f"{c.replace('residue-perm-','rp'):>16}" for c in CANDS)
          + f"{'argmin':>10}")
    for N in (1024, 2048, 4096):
        pieces = [Iv(0, 256), Ap(0, 8, N // 8)]
        cs = {c: oracle_cost(pieces, N, BQ, A, c) for c in CANDS}
        best = min((v, c) for c, v in cs.items() if v is not None)
        print(f"{N:>7}" + "".join(f"{cs[c]:>16,}" for c in CANDS)
              + f"{best[1].replace('residue-perm-','rp'):>10}")
