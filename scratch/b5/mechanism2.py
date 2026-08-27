import numpy as np
from polyattn import cost, transforms
from polyattn.explore import Custom, ZOO
TILES = [128, 64, 32, 16]

def closed_form_per_row(D, BQ, A):
    g = np.gcd(BQ, A); D = np.asarray(sorted(D))
    ms = np.arange(((D.min()-(A-1))//g)*g, D.max()+BQ-1+g, g)
    return g*int(sum(np.any((D >= m-BQ+1) & (D <= m+A-1)) for m in ms))

def measured_per_row(D, N, BQ, A):
    Dl = list(D)
    m = Custom(lambda q, kv: np.isin(q-kv, Dl), "d")
    lo, hi = N//2, 3*N//4          # interior only: every offset in D is realisable
    acc = 0
    for q0 in range(lo, hi, BQ):
        u = m.union_cols(q0, q0+BQ, N)
        acc += BQ*A*int(u.reshape(-1, A).any(axis=1).sum())
    return acc/(hi-lo)

print("=== closed form vs measured, N=8192 (interior rows only) ===")
print(f"{'D':<26}{'BQ':>5}{'A':>5}{'closed':>9}{'measured':>10}")
for D, lbl in [(list(range(128))+list(range(1024,1152)), "band0-128 + band1024"),
               (list(range(128))+list(range(1000,1128)), "band0-128 + band1000")]:
    for BQ, A in [(128,128),(128,16),(16,128),(32,64),(16,16)]:
        c, m_ = closed_form_per_row(D,BQ,A), measured_per_row(D,8192,BQ,A)
        print(f"{lbl:<26}{BQ:>5}{A:>5}{c:>9.1f}{m_:>10.1f}"
              + ("" if abs(c-m_)<1e-9 else "   <-- MISMATCH"))

print("\n=== C2: argmin transform within a max-class ===")
def mk(off, w, s, name):
    return Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < w) | (((q-kv) >= off) & ((q-kv) < off+w)) | ((q-kv) % s == 0)), name)
cases = [ZOO["local256+str8"],
         mk(1024, 256, 8, "str8 + band@1024(aligned)"),
         mk(1000, 256, 8, "str8 + band@1000(misaligned)"),
         mk(1004, 252, 8, "str8 + band@1004w252(misaligned)")]
for m in cases:
    N = 2048; M = m.dense(N); live = int(M.sum())
    variants = [(n, fn(M)[0]) for n, fn in transforms.candidates()
                if fn(M)[1] and fn(M)[1][0] == "A"]
    print(f"\n{m.name}")
    print("  BQ\\A  " + "".join(f"{a:>15}" for a in TILES))
    cells = {}
    for bq in TILES:
        row = []
        for a in TILES:
            w, n = min((transforms.tile_stats(Mt,bq,a)[1]/live, n) for n, Mt in variants)
            cells[(bq,a)] = (n, w); row.append(f"{n.replace('residue-perm-','rp')}({w:.2f})")
        print(f"  {bq:>4}  " + "".join(f"{c:>15}" for c in row))
    for mx in TILES:
        ns = {cells[(bq,a)][0] for bq in TILES for a in TILES if max(bq,a)==mx}
        ws = [cells[(bq,a)][1] for bq in TILES for a in TILES if max(bq,a)==mx]
        if len(ns) > 1 or max(ws)-min(ws) > 1e-9:
            print(f"    max={mx}: argmins={sorted(ns)}  waste spread={max(ws)-min(ws):.4f}"
                  + ("   <-- ARGMIN SPLITS WITHIN MAX-CLASS" if len(ns)>1 else ""))
