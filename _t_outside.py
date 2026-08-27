"""Rule 5 applied to my own suites, proactively.

What every one of my test inputs shared, that I never wrote down:
  A. displacement sets non-negative (causal). Only ONE signed case, never with transforms.
  B. N a power of two, for the transform suite (512, 1024 only).
  C. sinks always had g < w.
  D. docpack documents always longer than a tile.
  E. N always divisible by the fold depth s.
Each is varied here.
"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES as DC, cost_of as dcost
import general as G
from oracle import oracle_cost, tiles_cost, apply_xform

TIL = [128, 32, 16]
tot = bad = 0
fails = {}

def chk(tag, got, exp):
    global tot, bad
    if got is None:
        return
    tot += 1
    if got != exp:
        bad += 1
        fails.setdefault(tag, []).append((got, exp))

print("A. SIGNED (non-causal) displacement sets, with every transform")
for pieces, nm in [([Iv(-64, 64)], "signed-band-64"),
                   ([Iv(-200, -100), Iv(0, 50)], "signed-two-sided"),
                   ([Ap(-128, 8, 32)], "signed-lattice")]:
    for N in (512, 1024):
        for c in DC:
            for BQ in TIL:
                for A in TIL:
                    chk(f"A:{nm}:{c}", dcost(c, DiagSpec(pieces), N, BQ, A),
                        oracle_cost(pieces, N, BQ, A, c))

print("B. NON-POWER-OF-TWO N, transform suite")
for pieces, nm in [([Iv(0, 128)], "band128"), ([Iv(0, 32), Iv(100, 132)], "twoband-mis"),
                   ([Ap(0, 8, 96)], "lattice-8")]:
    for N in (768, 1536, 1152):
        for c in DC:
            for BQ in TIL:
                for A in TIL:
                    if N % BQ or N % A:
                        continue
                    chk(f"B:{nm}:{c}", dcost(c, DiagSpec(pieces), N, BQ, A),
                        oracle_cost(pieces, N, BQ, A, c))

print("C. sinks with g > w, and g comparable to N")
def dsink(g, w, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & ((kv < g) | (q - kv < w))
for g_, w_ in [(64, 8), (256, 16), (512, 4), (8, 8)]:
    for N in (1024, 1536):
        M0 = dsink(g_, w_, N)
        sp = G.Sinks(g_, w_)
        for c in G.CANDIDATES:
            Mt = apply_xform(M0, c)
            if Mt is None:
                continue
            for BQ in TIL:
                for A in TIL:
                    if N % BQ or N % A:
                        continue
                    chk(f"C:sinks{g_}+w{w_}:{c}", G.cost_of(c, sp, N, BQ, A),
                        tiles_cost(Mt, BQ, A))

print("D. docpack with documents SHORTER than a tile")
def ddoc(b, N):
    bb = np.array(list(b) + [N])
    d = np.searchsorted(bb, np.arange(N), side="right") - 1
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & (d[:, None] == d[None, :])
for b, nm in [(list(range(0, 1024, 8)), "docs-of-8"),
              ([0, 3, 5, 900], "tiny-docs"),
              (list(range(0, 1024, 100)), "docs-of-100")]:
    N = 1024
    M0 = ddoc(b, N); sp = G.DocPack(b, nm)
    for c in G.CANDIDATES:
        Mt = apply_xform(M0, c)
        if Mt is None:
            continue
        for BQ in TIL:
            for A in TIL:
                chk(f"D:{nm}:{c}", G.cost_of(c, sp, N, BQ, A), tiles_cost(Mt, BQ, A))

print("E. N NOT divisible by the fold depth s")
for N in (1000, 1500):
    for g_, w_ in [(4, 256)]:
        M0 = dsink(g_, w_, N); sp = G.Sinks(g_, w_)
        for c in G.CANDIDATES:
            Mt = apply_xform(M0, c)
            if Mt is None:
                continue
            for BQ in (100, 50) if N == 1000 else (100, 50):
                for A in (100, 50):
                    if N % BQ or N % A:
                        continue
                    chk(f"E:N{N}:{c}", G.cost_of(c, sp, N, BQ, A), tiles_cost(Mt, BQ, A))

print(f"\noutside-the-regime suite: {tot-bad}/{tot} exact")
for tag, v in sorted(fails.items())[:12]:
    print(f"  FAIL {tag:<34}{len(v):>4} cells, e.g. got {v[0][0]} want {v[0][1]}"
          f"  ({v[0][0]/max(1,v[0][1]):.3f}x)")
