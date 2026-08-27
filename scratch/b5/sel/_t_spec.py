"""Check DiagSpec.cost against a materialised reference. Independent of polyattn."""
import numpy as np, random
from blocks import Iv, Ap
from spec import DiagSpec

def ref_cost(pieces, N, BQ, A):
    d = np.arange(N)[:, None] - np.arange(N)[None, :]
    M = np.zeros((N, N), bool)
    for p in pieces:
        if isinstance(p, Iv):
            M |= (d >= p.lo) & (d < p.hi)
        else:
            for m in range(p.count):
                M |= (d == p.start + m * p.stride)
    M &= (d >= 0) | (d < 0)          # full square; D may include negative offsets
    t = M.reshape(N//BQ, BQ, N//A, A).any(axis=(1, 3))
    return int(t.sum()) * BQ * A, int(M.sum())

rng = random.Random(11)
CASES = [
    ("band128",        [Iv(0, 128)]),
    ("causal",         [Iv(0, 512)]),
    ("twoband-al",     [Iv(0, 32), Iv(128, 160)]),
    ("twoband-mis",    [Iv(0, 32), Iv(100, 132)]),
    ("twodiag-0-17",   [Iv(0, 1), Iv(17, 18)]),
    ("lattice-8",      [Ap(0, 8, 64)]),
    ("lattice-3",      [Ap(0, 3, 170)]),
    ("band+lat",       [Iv(0, 64), Ap(0, 8, 64)]),
    ("signed band",    [Iv(-48, 48)]),
]
bad = 0; n = 0
for name, pieces in CASES:
    for N in (256, 512):
        for BQ in (128, 64, 32, 16):
            for A in (128, 64, 32, 16):
                got = DiagSpec(pieces).cost(N, BQ, A)
                exp, live = ref_cost(pieces, N, BQ, A)
                n += 1
                if got != exp:
                    bad += 1
                    if bad <= 6:
                        print(f"MISMATCH {name} N={N} {BQ}x{A}: got {got} want {exp}")
print(f"DiagSpec.cost: {n-bad}/{n} exact" + ("" if not bad else "   <-- BROKEN"))
