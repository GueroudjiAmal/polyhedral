"""Which quantities in my code have a DIRECT test, versus only being exercised
through something else? Enumerated, then the untested ones tested.

  count_blocks   direct, 4000/4000                                   TESTED
  DiagSpec.cost  direct, 288/288 vs materialised reference           TESTED
  cost_of(...)   direct, 2400/2400 vs oracle                         TESTED
  general costs  direct, 1120+504+896 vs oracle                      TESTED
  live()         checked against ONE NOTES number, one mask, one N   -> here
  billed_cols    only via class-B costs                              -> here
  max_le         only via shear                                      -> here
  n_of_v         only via DiagSpec.cost                              -> here
  residue_runs   ??? possibly dead code                              -> here
"""
import sys, random
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
from blocks import Iv, Ap, count_blocks, billed_cols
from spec import DiagSpec, n_of_v
import xform, spec as specmod

rng = random.Random(5)
fail = 0

# --- live() -----------------------------------------------------------------
def ref_live(pieces, N):
    d = np.arange(N)[:, None] - np.arange(N)[None, :]
    M = np.zeros((N, N), bool)
    for p in pieces:
        if isinstance(p, Iv):
            M |= (d >= p.lo) & (d < p.hi)
        elif p.count > 0:
            last = p.start + (p.count - 1) * p.stride
            M |= (d >= p.start) & (d <= last) & ((d - p.start) % p.stride == 0)
    return int(M.sum())

bad = n = 0
CASES = [[Iv(0, 256), Ap(0, 8, 64)],            # OVERLAPPING -- the bug case
         [Iv(0, 128)], [Ap(0, 8, 64)], [Iv(0, 64), Iv(60, 200)],
         [Iv(0, 32), Ap(0, 3, 100)], [Iv(-40, 40)], [Ap(0, 5, 50), Ap(0, 7, 50)],
         [Iv(0, 100), Iv(100, 200), Ap(0, 2, 150)]]
for pieces in CASES:
    for N in (256, 512):
        n += 1
        got, exp = DiagSpec(pieces).live(N), ref_live(pieces, N)
        if got != exp:
            bad += 1
            print(f"  live() FAIL {pieces} N={N}: {got} vs {exp}")
print(f"live() direct test: {n-bad}/{n} exact"); fail += bad

# --- billed_cols ------------------------------------------------------------
bad = n = 0
for _ in range(3000):
    W = rng.randint(1, 300); A = rng.choice([1, 2, 4, 8, 16, 32])
    pieces, ref = [], set()
    for _ in range(rng.randint(1, 3)):
        lo = rng.randint(-10, W); hi = lo + rng.randint(0, 80)
        pieces.append(Iv(lo, hi)); ref |= set(range(max(0, lo), min(W, hi)))
    blocks = {x // A for x in ref}
    exp = sum(min(A, W - b * A) for b in blocks)
    got = billed_cols(pieces, A, W)
    n += 1
    if got != exp:
        bad += 1
        if bad <= 3:
            print(f"  billed_cols FAIL A={A} W={W} {pieces}: {got} vs {exp}")
print(f"billed_cols direct test: {n-bad}/{n} exact"); fail += bad

# --- max_le -----------------------------------------------------------------
bad = n = 0
for _ in range(2000):
    pieces = []
    ref = set()
    for _ in range(rng.randint(1, 3)):
        if rng.random() < .5:
            lo = rng.randint(0, 200); hi = lo + rng.randint(0, 60)
            pieces.append(Iv(lo, hi)); ref |= set(range(lo, hi))
        else:
            st = rng.randint(0, 100); sd = rng.randint(1, 20); c = rng.randint(0, 20)
            pieces.append(Ap(st, sd, c)); ref |= {st + m * sd for m in range(c)}
    x = rng.randint(-5, 260)
    got = DiagSpec(pieces).max_le(x)
    cand = [d for d in ref if d <= x]
    exp = max(cand) if cand else None
    n += 1
    if got != exp:
        bad += 1
        if bad <= 3:
            print(f"  max_le FAIL x={x} {pieces}: {got} vs {exp}")
print(f"max_le direct test: {n-bad}/{n} exact"); fail += bad

# --- n_of_v -----------------------------------------------------------------
bad = n = 0
for _ in range(2000):
    N = rng.choice([256, 512]); BQ = rng.choice([16, 32, 64, 128]); A = rng.choice([16, 32, 64, 128])
    v = rng.randint(-N, N)
    exp = sum(1 for i in range(N // BQ) for j in range(N // A) if j * A - i * BQ == v)
    got = n_of_v(v, N, BQ, A)
    n += 1
    if got != exp:
        bad += 1
        if bad <= 3:
            print(f"  n_of_v FAIL v={v} N={N} {BQ}x{A}: {got} vs {exp}")
print(f"n_of_v direct test: {n-bad}/{n} exact"); fail += bad

# --- dead code check --------------------------------------------------------
import subprocess
hits = subprocess.run(["grep", "-rn", "residue_runs",
                       "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/"],
                      capture_output=True, text=True).stdout.strip().splitlines()
callers = [h for h in hits if "def residue_runs" not in h and "_t_quantities" not in h]
print(f"\nresidue_runs: {len(callers)} caller(s) outside its definition"
      + ("   -> DEAD CODE, untested and unused" if not callers else ""))
print(f"\nTOTAL FAILURES: {fail}")
