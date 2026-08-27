"""Does the shape library's missing offset-band family distort experiment 3?

d4 audited shapes.LIBRARY: every band is anchored at diagonal offset 0, so a
union containing a band at a nonzero offset cannot be decomposed naturally --
the search falls back on the full-causal shape. That makes the misaligned case
UNTESTED, not tested-and-fine. This adds the family and re-runs, checking BOTH
the misaligned masks and the aligned ones d4's 1.19x came from.
"""
import numpy as np
from polyattn import shapes, transforms
from polyattn.experiments import compose
from polyattn.explore import Custom, ZOO

BASE = list(shapes.LIBRARY)


class OBand(shapes.Shape):
    """Causal band of width w at diagonal offset `off`:  off <= q-kv < off+w."""
    def __init__(self, off, w):
        # p stays SCALAR so compose.evaluate's `-(s.p or 0)` peel-order key works;
        # dense() is overridden because Shape's cache key is (kind, p, N) and would
        # otherwise collide across offsets. Both are noted in the diff to d4.
        self.kind, self.p = "oband", w
        self.off, self.w = off, w
        self.name = f"oband-{off}+{w}"

    def dense(self, N):
        key = ("oband", self.off, self.w, N)
        if key not in shapes._DENSE_CACHE:
            shapes._DENSE_CACHE[key] = self._dense(N)
        return shapes._DENSE_CACHE[key]

    def _dense(self, N):
        q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
        d = q - kv
        return (kv <= q) & (d >= self.off) & (d < self.off + self.w)

    def basis(self):
        return None


shapes.ORDER["oband"] = 0          # same tier as band: densest home first

MASKS = {
    "local256+str8 (zoo)": ZOO["local256+str8"],
    "str8+band@384 (aligned)": Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < 128) | (((q-kv) >= 384) & ((q-kv) < 512)) | ((q-kv) % 8 == 0)),
        "str8+band@384"),
    "str8+band@300 (misaligned)": Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < 128) | (((q-kv) >= 300) & ((q-kv) < 428)) | ((q-kv) % 8 == 0)),
        "str8+band@300"),
    "twoband@300 (no lattice)": Custom(lambda q, kv: (kv <= q) & (
        ((q-kv) < 128) | (((q-kv) >= 300) & ((q-kv) < 428))), "twoband@300"),
}

LIBS = {
    "current (offset-0 bands only)": BASE,
    "+ aligned obands (128-grid)": BASE + [OBand(o, w) for o in (128, 256, 384)
                                           for w in (128,)],
    "+ matched obands": BASE + [OBand(o, w) for o in (128, 256, 300, 384)
                                for w in (128,)],
}

print(f"{'mask':<28}{'library':<32}{'k=1':>8}{'best':>8}{'k':>3}  decomposition")
for mname, m in MASKS.items():
    for lname, lib in LIBS.items():
        shapes.LIBRARY = lib
        res, live = compose.search(m, N=1024, kmax=3, grain=(16, 16))
        if not res:
            print(f"{mname:<28}{lname:<32}{'-':>8}{'no exact cover':>8}")
            continue
        k1 = min((r["waste"] for r in res if r["k"] == 1), default=float("nan"))
        b = res[0]
        parts = " + ".join(f"{n}[{bs}]" for n, bs, _, _ in b["detail"])
        print(f"{mname:<28}{lname:<32}{k1:>8.3f}{b['waste']:>8.3f}{b['k']:>3}  {parts}")
    print()
shapes.LIBRARY = BASE
