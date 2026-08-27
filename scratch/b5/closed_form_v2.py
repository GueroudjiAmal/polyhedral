"""Resolving 2f's bounded-support gap.

TWO SEPARATE OBJECTS, which I conflated in my first message:

(1) SYMMETRY THEOREM.  My original proof routed through the counting formula and
    therefore inherited its bounded-D hypothesis.  Replacement proof needs no
    counting and no boundedness:

      T_D(BQ,A)  = # live tiles, mask live(q,kv) <=> q-kv in D, on [0,N)^2.
      (a) TRANSPOSE (q,kv)->(kv,q):  maps D -> -D and tile shape (BQ,A)->(A,BQ),
          so  T_D(BQ,A) = T_{-D}(A,BQ).
      (b) POINT REFLECTION (q,kv)->(N-1-q, N-1-kv):  maps D -> -D and, provided
          BQ | N and A | N, permutes row-blocks i -> N/BQ-1-i and column-blocks
          j -> N/A-1-j.  A bijection on tiles, so T_{-D}(A,BQ) = T_D(A,BQ).
      Compose:  T_D(BQ,A) = T_D(A,BQ).  Cost is BQ*A*T in both, so cost is equal.
    No hypothesis on D at all.  Holds for causal, dilated, anything.

(2) CLOSED FORM.  My stated version assumed every query block sees the same
    number of live tiles -- true only for bounded D.  Exact general version
    weights each diagonal offset v by how many tile pairs realise it:

      cost = BQ*A * sum_{v in g*Z, D cap [v-BQ+1, v+A-1] nonempty} n(v)
      n(v) = #{x : x = 0 mod BQ, x = -v mod A, max(0,-v) <= x < min(N, N-v)}
    and n(v) -> N/lcm(BQ,A) uniformly only when |v| << N, which recovers my
    g*|g*Z cap F| with relative error O(max|D| / N).  That is 2f's 1.016 ->
    1.008 -> 1.004, and their exact 2.000 for triangular support.
"""
import numpy as np
from polyattn import cost
from polyattn.explore import Custom, ZOO
TILES = [128, 64, 32, 16]


def n_of_v(v, N, BQ, A):
    g = np.gcd(BQ, A)
    if v % g:
        return 0
    lo, hi = max(0, -v), min(N, N - v)          # x = i*BQ must lie here
    if hi <= lo:
        return 0
    L = BQ * A // g
    # smallest x >= lo with x = 0 mod BQ and x = -v mod A
    x0 = None
    for x in range(lo + (-lo) % BQ, min(lo + L, hi), BQ):
        if (x + v) % A == 0:
            x0 = x
            break
    if x0 is None:
        return 0
    return (hi - 1 - x0) // L + 1


def exact_reorg(D, N, BQ, A):
    """Exact element count per query row, weighted -- no boundedness assumed."""
    D = np.asarray(sorted(D))
    g = np.gcd(BQ, A)
    lo, hi = D.min() - (A - 1), D.max() + (BQ - 1)
    tot = 0
    for v in range((lo // g) * g, hi + g, g):
        if np.any((D >= v - BQ + 1) & (D <= v + A - 1)):
            tot += n_of_v(v, N, BQ, A)
    return BQ * A * tot / N


def naive(D, N, BQ, A):
    """My first-message form: g * |gZ cap F|.  Bounded-D approximation."""
    D = np.asarray(sorted(D)); g = np.gcd(BQ, A)
    lo, hi = D.min() - (A - 1), D.max() + (BQ - 1)
    return g * sum(bool(np.any((D >= v - BQ + 1) & (D <= v + A - 1)))
                   for v in range((lo // g) * g, hi + g, g))


def measured(D, N, BQ, A):
    Dl = list(D)
    m = Custom(lambda q, kv: np.isin(q - kv, Dl), "d")
    return cost.cost(m, N, BQ, A, exact_only=True)[0] / N


CASES = {
    "band128 (bounded)":      lambda N: list(range(128)),
    "twoband-aligned":        lambda N: list(range(128)) + list(range(1024, 1152)),
    "twoband-misaligned":     lambda N: list(range(128)) + list(range(1000, 1128)),
    "causal (UNBOUNDED)":     lambda N: list(range(N)),
    "dilated-8 (UNBOUNDED)":  lambda N: list(range(0, N, 8)),
    "local256+str8 (UNBD)":   lambda N: sorted(set(range(256)) | set(range(0, N, 8))),
}

print("=== exact weighted form vs measured (ratio; 1.000 = exact) ===")
print(f"{'mask':<24}{'N':>6}" + "".join(f"{f'{b}x{a}':>12}" for b, a in
                                        [(128,128),(128,16),(16,128),(32,64),(16,16)]))
for name, mk in CASES.items():
    for N in (2048, 4096):
        D = mk(N)
        row = "".join(f"{exact_reorg(D,N,b,a)/measured(D,N,b,a):>12.6f}"
                      for b, a in [(128,128),(128,16),(16,128),(32,64),(16,16)])
        print(f"{name:<24}{N:>6}{row}")

print("\n=== my FIRST-MESSAGE form vs measured (2f's finding reproduced) ===")
print(f"{'mask':<24}{'N':>6}{'128x128':>12}{'128x16':>12}")
for name, mk in CASES.items():
    for N in (2048, 4096, 8192):
        D = mk(N)
        print(f"{name:<24}{N:>6}"
              f"{naive(D,N,128,128)/measured(D,N,128,128):>12.4f}"
              f"{naive(D,N,128,16)/measured(D,N,128,16):>12.4f}")

print("\n=== SYMMETRY as EXACT INTEGERS (not ratios), incl. unbounded D ===")
print(f"{'mask':<24}{'N':>6}  max |elems(BQ,A) - elems(A,BQ)| over all 16 cells")
allmasks = [(n, mk) for n, mk in CASES.items()] + \
           [("twodiag-0-17", lambda N: [0, 17]),
            ("prefix+causal", lambda N: list(range(N))),
            ("random-D", lambda N: sorted(np.random.default_rng(1)
                                          .choice(N, 300, replace=False).tolist()))]
for name, mk in allmasks:
    for N in (1024, 2048):
        D = mk(N); Dl = list(D)
        m = Custom(lambda q, kv: np.isin(q - kv, Dl), "d")
        e = {(b, a): round(cost.cost(m, N, b, a, exact_only=True)[0])
             for b in TILES for a in TILES}
        d = max(abs(e[(b, a)] - e[(a, b)]) for b in TILES for a in TILES)
        print(f"{name:<24}{N:>6}  {d}" + ("   <-- ASYMMETRY" if d else "   exact"))
