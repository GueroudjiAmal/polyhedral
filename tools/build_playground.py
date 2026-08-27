"""Generate notebooks/02_playground.ipynb -- a hands-on notebook, run once.

Unlike 01_reasoning_log.ipynb (regenerated from tools/build_notebook.py and never
hand-edited), the playground is a STARTING POINT you are meant to edit. Re-running
this script overwrites it, so copy it aside if you have work in there.
"""
import pathlib
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
def md(s): C.append(nbf.v4.new_markdown_cell(s.strip()))
def code(s): C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Playground — poke at each step by hand

**This notebook is yours to edit.** Every cell has a knob; change it and re-run.
(The read-only companion, `01_reasoning_log.ipynb`, is generated from
`tools/build_notebook.py` — edits there get overwritten. This one is generated
once by `tools/build_playground.py`, so copy it aside before re-running that.)

Everything runs in a second or two at these sizes. The question each step
answers:

| step | question |
|---|---|
| 1 | what does my mask look like, and is it well-formed? |
| 2 | how much work does a block mask waste on it? |
| 3 | does a change of basis help — and what does it cost? |
| 4 | which of the legal transforms is best? |
| 5 | would splitting the mask into parts do better? |
| 6 | how do I sweep a parameter and plot it? |

Background for any of it: [`docs/NOTES.md`](../docs/NOTES.md).
""")

code(r"""
%matplotlib inline
from polyattn.explore import *

sorted(ZOO)      # the built-in masks you can refer to by name
""")

md(r"""
## Step 1 — pick a mask, or write one

Two ways in. Either name one from `ZOO`, or write a predicate.

`Custom` takes `fn(q, kv) -> bool` evaluated on a meshgrid, so use numpy
operators — `&`, `|`, `~` — not Python's `and`/`or`. It derives everything
densely, which is exact but $O(N^2)$: keep `N` at a few thousand.

**Try changing the predicate below.** A few to start from:

```python
Custom(lambda q, kv: kv <= q,                              "causal")
Custom(lambda q, kv: (kv <= q) & (q - kv < 100),           "window-100")
Custom(lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0),      "every-8th")
Custom(lambda q, kv: (kv <= q) & ((q - kv < 64) | (kv < 4)), "sinks + window")
Custom(lambda q, kv: (kv <= q) & ((q // 128) == (kv // 128)), "block-diagonal")
```
""")

code(r"""
m = Custom(lambda q, kv: (kv <= q) & (q - kv < 100), "my-window")

validate(m)      # brute-force check -- always run this on a mask you wrote
""")

md(r"""
Draw it. Blue is live; **pink is computed but dead** — that is the waste a block
mask pays. `block=` is the tile size, `N=` the sequence length shown.
""")

code("show(m, N=384, block=64)")

md(r"""
## Step 2 — what does block granularity cost?

`waste_table` reports *elements computed / live elements* at each tile size.
`1.00` is perfect; `2.00` means half the work is thrown away.

The row that matters most is **128×128** — that is what FlexAttention's
`BlockMask` does today — against **16×16**, the floor tensor cores impose
(MMA is `m16n8k16`-shaped, so nothing finer is skippable in hardware).
""")

code("_ = waste_table(m, N=2048)")

md(r"""
**Worth trying:** shrink the window in step 1 to 32 and re-run. The 128×128 waste
climbs sharply while 16×16 barely moves — narrow bands are where finer bounds pay.
Then set the window to 2048 and watch the whole table collapse to 1.00: there is
nothing to win on a mask that is already dense.
""")

md(r"""
## Step 3 — try a change of basis

A transform re-indexes the iteration space. Only two families are legal, because
attention reduces over `kv` at fixed `q`:

| | transform | class | cost |
|---|---|---|---|
| keys | $kv' = \pi(kv)$ | **A** | free — permute K/V once per layer |
| keys | $kv' = (kv - aq)/s$ | **B** | per-tile gather, **cannot be amortised** |

`try_transform` reports *both* axes — FLOPs saved and traffic added — and calls
out the case where traffic grows faster than work shrinks.

Available: `identity`, `shear`, `residue-perm-{2,4,8}`, `stridefold-{2,4,8}`.
""")

code(r"""
strided = Custom(lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0), "every-8th")

try_transform(strided, "residue-perm-8", N=1024)
""")

md(r"""
That is the headline result: an 8× waste collapses to ~1.1 at **no traffic cost**,
because the permutation is independent of `q` and so can be applied to K/V once.

Now the counter-example. `shear` straightens a diagonal band into a rectangle and
drives waste to 1.00 — but it is class B, so every tile needs a different slice
of K/V:
""")

code('try_transform("window-128", "shear", N=1024)')

md(r"""
**Worth trying:** `try_transform(strided, "stridefold-8")`. It reaches exactly the
same waste as `residue-perm-8` and is useless anyway — same work, ~8× the traffic.
That pair is the clearest illustration of why the class matters more than the
element count.
""")

md("## Step 4 — search all the legal transforms")

code("_ = best_transform(strided, N=1024)")

md(r"""
Read the `cls` and `kv/tile` columns together. Rows with identical `waste` are not
equivalent: `residue-perm-8` and `stridefold-8` compute the same amount, but one
reads 16 kv rows per tile and the other 136.

**Worth trying:** run this on `"docpack-512"` (unaligned document boundaries) and
on `"causal"`. Neither has a transform that helps — for different reasons.
""")

md(r"""
## Step 5 — split the mask into parts

If a mask is a *union* of differently-structured families, no single basis suits
all of it. Split it into **disjoint** parts, give each its own basis, and merge
with the online-softmax (log-sum-exp) combine — the same trick flash-decoding
uses to split the kv axis.

Disjointness is not cosmetic: an element counted twice would be double-counted in
the softmax denominator.
""")

code('_ = split("local256+str8", N=512, kmax=2, top=6)')

md(r"""
The winner splits a mask that resisted every single transform (1.24) into a band
in the identity basis plus a lattice under `residue-perm-8`, reaching ~1.05.

**Worth trying:** `split("sinks4+win256")`. Splitting makes it *worse* — a 4-wide
sink prefix still occupies a full 16-wide tile, so peeling it off costs more than
it saves. A sub-mask narrower than the MMA granularity should never be split out.
""")

md(r"""
## Step 6 — your own experiment

A template: sweep one parameter, plot the three granularities. Change the family,
the range, or the granularities and re-run.
""")

code(r"""
import matplotlib.pyplot as plt
from polyattn import figures as F

N = 2048
widths = [16, 32, 64, 128, 256, 512, 1024]
grains = [(128, 128), (32, 32), (16, 16)]

series = {g: [] for g in grains}
for w in widths:
    mk = Custom(lambda q, kv, w=w: (kv <= q) & (q - kv < w), f"w{w}")
    live = mk.live_count(N)
    for g in grains:
        series[g].append(cost.cost(mk, N, *g, exact_only=True)[0] / live)

fig, ax = plt.subplots(figsize=(7.5, 4.4), facecolor=F.SURF)
ax.set_facecolor(F.SURF)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
for sp in ("left", "bottom"):
    ax.spines[sp].set_color(F.GRID)
ax.grid(True, color=F.GRID, lw=1); ax.set_axisbelow(True)
ax.tick_params(colors=F.INK2, labelsize=9, length=0)

for (g, ys), col in zip(series.items(), (F.S1, F.S2, F.S3)):
    ax.plot(widths, ys, marker="o", ms=5, lw=2, color=col, label=f"{g[0]}x{g[1]}")
ax.axhline(1.0, color=F.INK2, lw=1, ls=(0, (4, 3)))
ax.set_xscale("log", base=2); ax.set_xticks(widths)
ax.set_xticklabels(widths); ax.set_xlabel("window width", color=F.INK2, fontsize=9.5)
ax.set_ylabel("elements computed / live", color=F.INK2, fontsize=9.5)
ax.set_title("Boundary waste is a small-window phenomenon", loc="left",
             color=F.INK, fontsize=11.5, pad=10)
ax.legend(frameon=False, fontsize=9, labelcolor=F.INK2, title="tile size",
          title_fontsize=9)
fig.tight_layout()
""")

md(r"""
## Scratch

Yours. `from polyattn.explore import *` already gave you `masks`, `transforms`,
`compose` and `cost` if you want the lower-level API.

Useful entry points:

```python
cost.cost(m, N, BQ, A, exact_only=True)     # (elements computed, was_sampled)
transforms.tile_stats(M, BQ, A)             # (tiles, elements) on a dense array
transforms.candidates()                     # [(name, fn), ...]
compose.search(m, N=512, kmax=2)            # full decomposition search
m.dense(N)                                  # the raw N x N boolean mask
```
""")

code("")

nb["cells"] = C
nb.metadata.kernelspec = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = pathlib.Path(__file__).resolve().parents[1] / "notebooks" / "02_playground.ipynb"
nbf.write(nb, str(out))
print(f"wrote {out} ({len(C)} cells)")
