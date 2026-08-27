"""b5 adversarial check of C3: diagonal invariance => waste is a function of max(BQ,A).

Two halves:
  A) SYMMETRY  w(BQ,A) == w(A,BQ)  -- claimed provable, tested here.
  B) MAX-LAW   w constant within a max-class -- claimed FALSE in general;
     counterexamples constructed, with an aligned control that obeys it.
"""
import sys
import numpy as np
from polyattn import cost
from polyattn.explore import Custom, ZOO

TILES = [128, 64, 32, 16]


def grid(m, N):
    live = m.live_count(N)
    return {(bq, a): cost.cost(m, N, bq, a, exact_only=True)[0] / live
            for bq in TILES for a in TILES}


def report(m, N, expect=None):
    g = grid(m, N)
    sym = max(abs(g[(bq, a)] - g[(a, bq)]) for bq in TILES for a in TILES)
    print(f"\n=== {m.name}  N={N}  live={m.live_count(N):,} "
          f"(density {m.live_count(N)/N/N*100:.3f}%)")
    print("  BQ\\A  " + "".join(f"{a:>10}" for a in TILES))
    for bq in TILES:
        print(f"  {bq:>4}  " + "".join(f"{g[(bq,a)]:>10.4f}" for a in TILES))
    print(f"  transpose symmetry |w(BQ,A)-w(A,BQ)|max = {sym:.4f}")
    worst = 0.0
    for mx in TILES:
        cls = [g[(bq, a)] for bq in TILES for a in TILES if max(bq, a) == mx]
        spread = max(cls) - min(cls)
        worst = max(worst, spread)
        print(f"  max={mx:>3}: n={len(cls):>2}  min={min(cls):.4f}  "
              f"max={max(cls):.4f}  SPREAD={spread:.4f}")
    verdict = "MAX-LAW HOLDS" if worst < 1e-9 else f"MAX-LAW VIOLATED (spread {worst:.4f})"
    print(f"  --> {verdict}"
          + (f"   [expected: {expect}]" if expect else ""))
    return sym, worst


# ---- the mask family: causal, diagonally invariant, union of two bands -------
def two_band(off, w, name):
    return Custom(lambda q, kv: (kv <= q) & (
        ((q - kv) < w) | (((q - kv) >= off) & ((q - kv) < off + w))), name)


N = int(sys.argv[1]) if len(sys.argv) > 1 else 2048

# controls from their own zoo: should obey (single interval / sub-tile stride)
for nm in ("window-128", "dilated-8", "local256+str8"):
    if nm in ZOO:
        report(ZOO[nm], N, expect="holds (their result)")

# aligned control: both bands aligned to the coarsest tile -> predicted to HOLD
report(two_band(1024, 128, "twoband-aligned-1024"), N, expect="holds")
# same shape, offset shifted by 8 -> predicted to VIOLATE
report(two_band(1000, 128, "twoband-misaligned-1000"), N, expect="VIOLATES")
# minimal case: two diagonals, offsets 0 and 17
report(Custom(lambda q, kv: (kv <= q) & (((q - kv) == 0) | ((q - kv) == 17)),
              "twodiag-0-17"), N, expect="VIOLATES")
