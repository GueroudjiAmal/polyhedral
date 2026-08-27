"""(1) CORRECTED padding over-charge.  My first metric was wrong: zero-padding
adds no phantom TILES (an all-zero tile is never counted). The real over-charge
is that the final partial column strip is billed a full A columns when only
nk % A exist.  Charge = tiles * BQ * A regardless.

(2) The sharper version of d4's shape-library gap: offset BANDS changed nothing,
because `all` already catches any remainder in the identity basis and splitting a
part only costs more unless the parts want DIFFERENT bases.  So the family that
could actually bite is one whose natural home is a NON-identity basis and which
no library shape expresses: a lattice confined to an offset window.
"""
import numpy as np
from polyattn import masks, transforms, shapes
from polyattn.experiments import compose
from polyattn.explore import Custom

print("=== (1) real over-charge from billing the final partial column strip ===")
print(f"{'mask':<18}{'transform':<14}{'shape':>14}{'billed':>12}{'occupied':>12}{'over':>8}")
rows = 0
for m in masks.zoo():
    M = m.dense(4096)
    for tname, fn in transforms.candidates():
        Mt, _ = fn(M)
        if Mt is None:
            continue
        nq, nk = Mt.shape
        A = BQ = 16
        if nk % A == 0 and nq % BQ == 0:
            continue
        tiles = transforms.tile_stats(Mt, BQ, A)[0]
        billed = tiles * BQ * A
        # tiles in the final partial column strip, each over-billed by (A - nk%A) cols
        Pq = (-nq) % BQ
        Mp = np.pad(Mt, ((0, Pq), (0, (-nk) % A)))
        t = Mp.reshape(-1, BQ, Mp.shape[1] // A, A).any(axis=(1, 3))
        last = int(t[:, -1].sum()) if nk % A else 0
        over = last * BQ * (A - nk % A) if nk % A else 0
        over += (int(t.sum()) - last) * 0
        if over:
            rows += 1
            print(f"{m.name:<18}{tname:<14}{str(Mt.shape):>14}{billed:>12,}"
                  f"{billed-over:>12,}{over/billed*100:>7.3f}%")
print(f"  affected transform outputs: {rows}   (all are class-B shear/stridefold;"
      f" class-A permutations keep the matrix square)")

print("\n=== (2) does a MISSING NON-IDENTITY-BASIS shape change the answer? ===")
BASE = list(shapes.LIBRARY)


class OLattice(shapes.Shape):
    """stride-s lattice confined to off <= q-kv < off+w.  Wants residue-perm-s;
    no shape in the current library expresses it, so it lands in `all` at identity."""
    def __init__(self, off, w, s):
        self.kind, self.p, self.off, self.w, self.s = "olattice", s, off, w, s
        self.name = f"olattice-{off}+{w}/{s}"

    def dense(self, N):
        key = ("olat", self.off, self.w, self.s, N)
        if key not in shapes._DENSE_CACHE:
            q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]; d = q - kv
            shapes._DENSE_CACHE[key] = ((kv <= q) & (d >= self.off)
                                        & (d < self.off + self.w) & (d % self.s == 0))
        return shapes._DENSE_CACHE[key]

    def basis(self):
        return transforms.make_residue_perm(self.s)


shapes.ORDER["olattice"] = 1

TESTS = {
    # band + a REMOTE STRIDED window: the remote piece wants rp4, nothing expresses it
    "band128 + lat4@1024w512": Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < 128) | (((q-kv) >= 1024) & ((q-kv) < 1536) & ((q-kv) % 4 == 0))),
        "band128+lat4@1024"),
    "band128 + lat4@1000w512": Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < 128) | (((q-kv) >= 1000) & ((q-kv) < 1512) & ((q-kv) % 4 == 0))),
        "band128+lat4@1000"),
}
LIBS = {"current": BASE,
        "+ olattice(1024,512,4)": BASE + [OLattice(1024, 512, 4)],
        "+ olattice(1000,512,4)": BASE + [OLattice(1000, 512, 4)]}
print(f"{'mask':<26}{'library':<26}{'k=1':>8}{'best':>8}{'k':>3}  decomposition")
for mn, m in TESTS.items():
    for ln, lib in LIBS.items():
        shapes.LIBRARY = lib
        res, live = compose.search(m, N=2048, kmax=3, grain=(16, 16))
        if not res:
            print(f"{mn:<26}{ln:<26}{'no cover':>8}"); continue
        k1 = min((r["waste"] for r in res if r["k"] == 1), default=float("nan"))
        b = res[0]
        parts = " + ".join(f"{n}[{bs}]" for n, bs, _, _ in b["detail"])
        print(f"{mn:<26}{ln:<26}{k1:>8.3f}{b['waste']:>8.3f}{b['k']:>3}  {parts}")
    print()
shapes.LIBRARY = BASE
