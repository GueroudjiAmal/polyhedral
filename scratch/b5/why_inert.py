"""My prediction failed: olattice was inert too. Find the actual reason, then
test the reason rather than assert it.

HYPOTHESIS 2: every shape is INTERSECTED with M before use --
compose.search does inter[sh.name] = sh.dense(N) & M. So a GLOBAL shape
automatically specialises to whichever part of M it overlaps: the offset comes
from M, not from the shape. Under that hypothesis no offset variant of an
existing shape can ever add anything, and the only unreachable pieces are ones
that cannot be written as (library shape & M) minus earlier peels.

FALSIFIABLE CONSEQUENCE: a mask needing TWO DIFFERENT strides in two different
windows should defeat the current library, because lattice-8 & M grabs elements
from both windows (d%8==0 implies d%4==0), scattering them into one part that
can hold only one basis. olattice shapes should then fix it.
If this is ALSO inert, hypothesis 2 is wrong too.
"""
import numpy as np
from polyattn import shapes, transforms
from polyattn.experiments import compose
from polyattn.explore import Custom

BASE = list(shapes.LIBRARY)


class OLattice(shapes.Shape):
    def __init__(self, off, w, s):
        self.kind, self.p, self.off, self.w, self.s = "olattice", s, off, w, s
        self.name = f"olat-{off}+{w}/{s}"

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

# two different strides in two disjoint diagonal windows
m = Custom(lambda q, kv: (kv <= q) & (
    ((q - kv) < 128)
    | (((q - kv) >= 512) & ((q - kv) < 1024) & ((q - kv) % 4 == 0))
    | (((q - kv) >= 1024) & ((q - kv) < 1536) & ((q - kv) % 8 == 0))),
    "band128 + lat4@512 + lat8@1024")

print("first, confirm hypothesis 2's premise directly:")
N = 2048
M = m.dense(N)
for sh in BASE:
    if sh.kind == "lattice":
        inter = sh.dense(N) & M
        d = (np.arange(N)[:, None] - np.arange(N)[None, :])
        w1 = int((inter & (d >= 512) & (d < 1024)).sum())
        w2 = int((inter & (d >= 1024) & (d < 1536)).sum())
        print(f"  {sh.name} & M: {int(inter.sum()):>8} elems"
              f"   from lat4 window {w1:>7}   from lat8 window {w2:>7}"
              + ("   <-- MIXES the two windows" if w1 and w2 else ""))

LIBS = {"current": BASE,
        "+ olat(512,512,4)": BASE + [OLattice(512, 512, 4)],
        "+ both olattices": BASE + [OLattice(512, 512, 4), OLattice(1024, 512, 8)]}
print(f"\n{'library':<24}{'k=1':>8}{'best':>8}{'k':>3}  decomposition")
for ln, lib in LIBS.items():
    shapes.LIBRARY = lib
    res, live = compose.search(m, N=2048, kmax=3, grain=(16, 16))
    if not res:
        print(f"{ln:<24}{'NO EXACT COVER':>8}"); continue
    k1 = min((r["waste"] for r in res if r["k"] == 1), default=float("nan"))
    b = res[0]
    print(f"{ln:<24}{k1:>8.3f}{b['waste']:>8.3f}{b['k']:>3}  "
          + " + ".join(f"{n}[{bs}]" for n, bs, _, _ in b["detail"]))
shapes.LIBRARY = BASE
