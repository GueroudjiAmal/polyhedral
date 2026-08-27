"""Brute-force check of count_blocks -- independent of the rest."""
import random
from blocks import Iv, Ap, count_blocks

rng = random.Random(7)
bad = 0
for trial in range(4000):
    ncols = rng.choice([64, 128, 256, 500])
    A = rng.choice([1, 2, 4, 8, 16, 32, 64])
    pieces, ref = [], set()
    for _ in range(rng.randint(1, 4)):
        if rng.random() < 0.5:
            lo = rng.randint(-20, ncols + 10); hi = lo + rng.randint(0, 90)
            pieces.append(Iv(lo, hi))
            ref |= {x for x in range(max(0, lo), min(ncols, hi))}
        else:
            st = rng.randint(-20, ncols); sd = rng.randint(1, 70); c = rng.randint(0, 40)
            pieces.append(Ap(st, sd, c))
            ref |= {st + m * sd for m in range(c) if 0 <= st + m * sd < ncols}
    got = count_blocks(pieces, A, ncols)
    exp = len({x // A for x in ref})
    if got != exp:
        bad += 1
        if bad <= 5:
            print(f"MISMATCH A={A} ncols={ncols} {pieces} got {got} want {exp}")
print(f"count_blocks: {4000-bad}/4000 exact" + ("" if not bad else "   <-- BROKEN"))
