"""Interactive helpers -- the API the playground notebook is built on.

Everything here is meant to be called by hand, edited, and re-run. It trades
speed for convenience: `Custom` derives its closed forms by brute force so you
can write a predicate and immediately push it through the whole pipeline, which
is fine up to N of a few thousand and hopeless beyond that.

    from polyattn.explore import *

    m = Custom(lambda q, kv: (kv <= q) & (q - kv < 100), "my-window")
    show(m)                       # draw it, with the block waste annotated
    waste_table(m)                # what it costs at each granularity
    try_transform(m, "shear")     # before/after, drawn and scored
    best_transform(m)             # search all legal transforms
    split(m)                      # search disjoint decompositions
"""
import numpy as np

from . import cost, figures, masks, shapes, transforms
from .experiments import compose

__all__ = ["Custom", "show", "waste_table", "try_transform", "best_transform",
           "split", "validate", "ZOO", "masks", "transforms", "compose", "cost"]

ZOO = {m.name: m for m in masks.zoo()}


class Custom(masks.Mask):
    """A mask from a predicate `fn(q, kv) -> bool`, evaluated on a meshgrid.

    Both arguments arrive as broadcast integer arrays, so write the predicate
    with numpy operators (`&`, `|`, `~`) rather than `and`/`or`:

        Custom(lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0), "every-8th")

    All the closed forms `masks.Mask` normally supplies are derived densely
    here, so this is exact but O(N^2). Keep N at a few thousand at most.
    """
    family = "custom"
    data_dependent = False

    def __init__(self, fn, name="custom"):
        self.fn, self.name = fn, name
        self._cache = {}

    def dense(self, N):
        if N not in self._cache:
            q = np.arange(N)[:, None]
            kv = np.arange(N)[None, :]
            M = np.asarray(self.fn(q, kv))
            self._cache[N] = np.broadcast_to(M, (N, N)).astype(bool)
        return self._cache[N]

    def row_cols(self, q, N):
        return self.dense(N)[q]

    def union_cols(self, q0, q1, N):
        return self.dense(N)[q0:q1].any(axis=0)

    def live_count(self, N):
        return int(self.dense(N).sum())


def _resolve(m):
    return ZOO[m] if isinstance(m, str) else m


def _transform(t):
    """Accept a transform by name or as a callable."""
    if callable(t):
        return t
    for name, fn in transforms.candidates():
        if name == t:
            return fn
    raise KeyError(f"unknown transform {t!r}; try {[n for n, _ in transforms.candidates()]}")


# ------------------------------------------------------------------ looking --
def show(m, N=384, block=64):
    """Draw one mask: live elements, and the tiles a block mask would compute."""
    return figures.fig_mask_gallery(N=N, block=block, specs=[_resolve(m)])


def waste_table(m, N=2048, grains=((128, 128), (64, 64), (32, 32), (16, 16))):
    """Elements computed / live, at each tile granularity. 1.00 is perfect."""
    m = _resolve(m)
    live = m.live_count(N)
    print(f"{m.name}  N={N}  live={live:,}  density={live/N/N*100:.2f}%\n")
    print(f"{'BQ x A':>10}{'computed':>16}{'waste':>9}")
    print("-" * 35)
    out = {}
    for BQ, A in grains:
        c, _ = cost.cost(m, N, BQ, A, exact_only=True)
        out[(BQ, A)] = c / live
        print(f"{f'{BQ}x{A}':>10}{c:>16,.0f}{c/live:>9.3f}")
    return out


# --------------------------------------------------------------- transforms --
def try_transform(m, t, N=1024, grain=(16, 16), draw=True, draw_n=384):
    """Apply one transform and report what it did to both FLOPs and traffic."""
    m = _resolve(m)
    fn = _transform(t)
    M = m.dense(N)
    live = int(M.sum())
    Mt, meta = fn(M)
    if Mt is None:
        print(f"{t}: not applicable to {m.name}")
        return None
    assert int(Mt.sum()) == live, "transform lost elements"
    kind, a, s = meta
    before = transforms.tile_stats(M, *grain)[1] / live
    after = transforms.tile_stats(Mt, *grain)[1] / live
    kv_before = transforms.kv_per_tile("A", 0, 1, *grain)
    kv_after = transforms.kv_per_tile(kind, a, s, *grain)
    print(f"{m.name}  --{t}-->   class {kind}"
          f"{'  (free: permute K/V once per layer)' if kind == 'A' else '  (per-tile gather, NOT amortisable)'}")
    print(f"  waste at {grain[0]}x{grain[1]}   {before:.3f}  ->  {after:.3f}"
          f"   ({before/after:.2f}x fewer elements)")
    print(f"  kv rows / tile      {kv_before}  ->  {kv_after}"
          f"   ({kv_after/kv_before:.2f}x traffic)")
    if kv_after / kv_before > before / after:
        print("  VERDICT: net loss -- traffic grows faster than work shrinks.")
    if draw:
        return _draw_pair(M, Mt, m.name, str(t), draw_n, grain[1])
    return dict(before=before, after=after, cls=kind,
                traffic=kv_after / kv_before)


def _draw_pair(M, Mt, name, tname, n, block):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap([figures.SURF, "#f7cdb9", figures.S1])
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 5.0), facecolor=figures.SURF)
    for ax, (A, title) in zip(axes, [(M, "before"), (Mt, f"after  {tname}")]):
        A = A[:n, :min(n, A.shape[1])]
        h, w = A.shape
        ph, pw = (-h) % block, (-w) % block
        P = np.pad(A, ((0, ph), (0, pw)))
        t = P.reshape(P.shape[0] // block, block, P.shape[1] // block, block).any(axis=(1, 3))
        img = np.where(np.kron(t, np.ones((block, block), bool)), 1, 0)
        img[:h, :w][A] = 2
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest", aspect="auto")
        live = int(A.sum())
        ax.set_title(f"{title}\nwaste {int(t.sum())*block*block/live:.2f}x",
                     fontsize=10, color=figures.INK, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(figures.GRID)
    fig.suptitle(f"{name}  -  top-left {n}x{n}, {block}x{block} tiles",
                 fontsize=12, color=figures.INK, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def best_transform(m, N=1024, grain=(16, 16)):
    """Score every legal transform. Free (class A) wins ties -- see NOTES §4."""
    m = _resolve(m)
    M = m.dense(N)
    live = int(M.sum())
    rows = []
    for name, fn in transforms.candidates():
        Mt, meta = fn(M)
        if Mt is None:
            continue
        kind, a, s = meta
        rows.append((transforms.tile_stats(Mt, *grain)[1] / live, kind,
                     transforms.kv_per_tile(kind, a, s, *grain), name))
    rows.sort(key=lambda r: (round(r[0], 4), r[1] != "A"))
    print(f"{m.name}  N={N}  grain {grain[0]}x{grain[1]}\n")
    print(f"{'transform':<18}{'cls':>4}{'waste':>9}{'kv/tile':>9}{'free?':>7}")
    print("-" * 47)
    for w, kind, kv, name in rows:
        print(f"{name:<18}{kind:>4}{w:>9.3f}{kv:>9}{'yes' if kind == 'A' else 'no':>7}")
    return rows


# -------------------------------------------------------------- composition --
def split(m, N=512, kmax=2, top=8, grain=(16, 16)):
    """Search disjoint decompositions -- each part free to pick its own basis.

    Small N and kmax=2 by default so this returns in seconds; the full search
    (polyattn.experiments.compose) uses N=1024, kmax=3 and takes minutes.
    """
    m = _resolve(m)
    res, live = compose.search(m, N=N, kmax=kmax, grain=grain)
    if not res:
        print(f"{m.name}: no decomposition in the library reconstructs it exactly")
        return []
    print(f"{m.name}  N={N}  live={live:,}  grain {grain[0]}x{grain[1]}\n")
    for r in res[:top]:
        parts = " + ".join(f"{n}[{b}]" for n, b, _, _ in r["detail"])
        print(f"  waste {r['waste']:.3f}   k={r['k']}   {parts}")
    return res


# ---------------------------------------------------------------- checking ---
def validate(m, N=256):
    """Brute-force check a mask's closed forms. Run this on anything you write."""
    m = _resolve(m)
    M = m.dense(N)
    problems = []
    if m.live_count(N) != int(M.sum()):
        problems.append(f"live_count {m.live_count(N)} != {int(M.sum())}")
    for BQ in (16, 64):
        for b in range(N // BQ):
            q0 = b * BQ
            if not np.array_equal(m.union_cols(q0, q0 + BQ, N), M[q0:q0 + BQ].any(axis=0)):
                problems.append(f"union_cols wrong at BQ={BQ} block={b}")
                break
    c, _ = cost.cost(m, N, 1, 1, exact_only=True)
    if c != int(M.sum()):
        problems.append(f"cost(1,1) {c} != live {int(M.sum())}")
    print(f"{m.name}: " + ("OK" if not problems else "PROBLEMS"))
    for p in problems:
        print("  -", p)
    return not problems
