"""b5: the closed form behind the max-law, plus N-dependence, plus C2 (selection).

Closed form (infinite/toroidal, diagonally-invariant mask with offset set D):
    elements computed per query row = g * |{ m in g*Z : (D + [-(A-1), BQ-1]) contains m }|
    where g = gcd(BQ, A) = min(BQ, A) for power-of-two tiles.
Consequences:
  - SYMMETRY is a theorem: F(A,BQ) = F(BQ,A) - (BQ-A), and BQ-A is a multiple of g,
    so m -> m-(BQ-A) is a bijection of g*Z carrying one count to the other.
  - MAX-LAW is NOT: it needs each interval of D to be g-aligned for every g in the
    tile set, and the dilated pieces not to merge differently at different g.
"""
import numpy as np
from polyattn import cost, transforms
from polyattn.explore import Custom, ZOO

TILES = [128, 64, 32, 16]


def closed_form_per_row(D, BQ, A):
    """g * #(g*Z intersect (D + [-(A-1), BQ-1]))."""
    g = np.gcd(BQ, A)
    D = np.asarray(sorted(D))
    lo, hi = D.min() - (A - 1), D.max() + (BQ - 1)
    ms = np.arange((lo // g) * g, hi + g, g)
    # m live iff some d in D with m-BQ+1 <= d <= m+A-1
    live = np.array([np.any((D >= m - BQ + 1) & (D <= m + A - 1)) for m in ms])
    return g * int(live.sum())


def measured_per_row(D, N, BQ, A):
    """Torus-free finite check: use a wide non-causal band-set well inside [0,N)."""
    Dset = set(D)
    m = Custom(lambda q, kv: np.isin(q - kv, list(Dset)), f"D{sorted(Dset)}")
    # measure only interior row-blocks so boundary truncation does not pollute
    lo, hi = N // 4, 3 * N // 4
    acc = 0
    for q0 in range(lo, hi, BQ):
        u = m.union_cols(q0, q0 + BQ, N)
        acc += BQ * A * int(u.reshape(-1, A).any(axis=1).sum())
    return acc / (hi - lo)


print("=== closed form vs measured (interior rows, N=1024) ===")
print(f"{'D':<22}{'BQ':>5}{'A':>5}{'closed':>9}{'measured':>10}")
for D in ([0], list(range(128)), [0, 17], list(range(128)) + list(range(1000, 1128))):
    lbl = f"{D[:2]}..{D[-1]}" if len(D) > 3 else str(D)
    for BQ, A in [(128, 128), (128, 16), (16, 128), (32, 64), (16, 16)]:
        c, m_ = closed_form_per_row(D, BQ, A), measured_per_row(D, 1024, BQ, A)
        flag = "" if abs(c - m_) < 1e-9 else "   <-- MISMATCH"
        print(f"{lbl:<22}{BQ:>5}{A:>5}{c:>9.1f}{m_:>10.1f}{flag}")

print("\n=== N-dependence of the violation (twoband-misaligned-1000) ===")
def two_band(off, w, name):
    return Custom(lambda q, kv: (kv <= q) & (
        ((q - kv) < w) | (((q - kv) >= off) & ((q - kv) < off + w))), name)

print(f"{'N':>6}{'mask':<26}{'sym':>8}{'spread@max=128':>16}")
for N in (2048, 4096, 8192):
    for m in (two_band(1024, 128, "aligned-1024"), two_band(1000, 128, "misaligned-1000")):
        live = m.live_count(N)
        g = {(bq, a): cost.cost(m, N, bq, a, exact_only=True)[0] / live
             for bq in TILES for a in TILES}
        sym = max(abs(g[(bq, a)] - g[(a, bq)]) for bq in TILES for a in TILES)
        cls = [g[(bq, a)] for bq in TILES for a in TILES if max(bq, a) == 128]
        print(f"{N:>6}{m.name:<26}{sym:>8.4f}{max(cls)-min(cls):>16.4f}")

print("\n=== C2: does the ARGMIN transform also break the max-law? ===")
for m in (two_band(1000, 128, "misaligned-1000"), ZOO["local256+str8"]):
    N = 2048
    M = m.dense(N)
    live = int(M.sum())
    variants = []
    for name, fn in transforms.candidates():
        Mt, meta = fn(M)
        if Mt is None or meta[0] != "A":
            continue
        variants.append((name, Mt))
    print(f"\n{m.name}  (class-A candidates: {[n for n,_ in variants]})")
    print("  BQ\\A  " + "".join(f"{a:>16}" for a in TILES))
    for bq in TILES:
        row = []
        for a in TILES:
            best = min(((transforms.tile_stats(Mt, bq, a)[1] / live), n) for n, Mt in variants)
            row.append(f"{best[1].replace('residue-perm-','rp')}({best[0]:.2f})")
        print(f"  {bq:>4}  " + "".join(f"{c:>16}" for c in row))
