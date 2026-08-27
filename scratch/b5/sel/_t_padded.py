"""Verify 2f's padded-extent biconditional MYSELF rather than relaying it.

2f's claim: cost symmetry w(BQ,A) == w(A,BQ) holds  <=>  ceil(N/BQ)*BQ == ceil(N/A)*A
i.e. the operative hypothesis is that the two PADDED extents agree, not that
BQ|N and A|N (my original, stronger statement), and not squareness.
"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from oracle import dense, tiles_cost

def pad(n, t): return -(-n // t) * t

MASKS = [("causal", lambda N: [Iv(0, N)]),
         ("window-128", lambda N: [Iv(0, 128)]),
         ("dilated-8", lambda N: [Ap(0, 8, max(1, N // 8))]),
         ("twoband-mis", lambda N: [Iv(0, 128), Iv(1000, 1128)])]
TIL = [128, 64, 32, 16]
tp = tn = fp = fn = 0
ex = []
for N in (500, 777, 900, 1000, 1023, 1200, 1536, 2048):
    for nm, mk in MASKS:
        M = dense(mk(N), N)
        for BQ in TIL:
            for A in TIL:
                asym = tiles_cost(M, BQ, A) - tiles_cost(M, A, BQ)
                pred = (pad(N, BQ) == pad(N, A))
                got = (asym == 0)
                if pred and got: tp += 1
                elif not pred and not got: tn += 1
                elif pred and not got:
                    fp += 1; ex.append(("PRED-SYM but ASYM", nm, N, BQ, A, asym))
                else:
                    fn += 1; ex.append(("PRED-ASYM but SYM", nm, N, BQ, A, asym))
print("2f's biconditional: symmetric <=> ceil(N/BQ)*BQ == ceil(N/A)*A")
print(f"  predicted symmetric AND symmetric      {tp}")
print(f"  predicted asymmetric AND asymmetric    {tn}")
print(f"  predicted symmetric BUT asymmetric     {fp}   <- would falsify")
print(f"  predicted asymmetric BUT symmetric     {fn}   <- would weaken")
print(f"  total cells {tp+tn+fp+fn}")
for e in ex[:8]:
    print("   ", e)
print(f"\nverdict: {'CONFIRMED both directions' if fp == 0 and fn == 0 else 'NOT confirmed'}")
