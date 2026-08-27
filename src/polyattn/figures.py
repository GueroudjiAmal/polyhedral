"""All figures. Import and call, or run as a script to write the PNGs.

Palette: categorical slots 1-3 of the validated default (blue / orange / aqua),
which is the set that passes the all-pairs CVD gate. Every series is also
direct-labelled, so identity is never carried by colour alone.
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from .paths import FIGURES

S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
S4, S5 = "#eda100", "#e87ba4"
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"


def _axis(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID); ax.spines["bottom"].set_linewidth(1)
    ax.xaxis.grid(True, color=GRID, lw=1); ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, length=0, labelsize=9)
    return ax


# ----------------------------------------------------------- mask gallery ----
def fig_mask_gallery(N=384, block=64, specs=None):
    """What 'block granularity' means, drawn. Three levels: dead / computed-but-
    dead / live, so the wasted work is the thing you actually see."""
    from . import masks
    specs = specs or [masks.SlidingWindow(64), masks.Dilated(4),
                      masks.DocPacked(96, seed=1)]
    cmap = ListedColormap([SURF, "#f7cdb9", S1])
    fig, axes = plt.subplots(1, len(specs), figsize=(4.1 * len(specs), 4.6),
                             facecolor=SURF, squeeze=False)
    axes = axes[0]
    for ax, m in zip(axes, specs):
        M = m.dense(N)
        tiles = M.reshape(N // block, block, N // block, block).any(axis=(1, 3))
        img = np.where(np.kron(tiles, np.ones((block, block), bool)), 1, 0)
        img[M] = 2
        ax.imshow(img, cmap=cmap, vmin=0, vmax=2, interpolation="nearest")
        for k in range(0, N + 1, block):
            ax.axhline(k - .5, color=GRID, lw=.6); ax.axvline(k - .5, color=GRID, lw=.6)
        live = int(M.sum()); comp = int(tiles.sum()) * block * block
        ax.set_title(f"{m.name}\nwaste at {block}x{block} = {comp/live:.2f}x",
                     fontsize=10, color=INK, pad=8)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    axes[0].set_ylabel("query  q", fontsize=9, color=INK2)
    for ax in axes:
        ax.set_xlabel("key  kv", fontsize=9, color=INK2)
    fig.legend(handles=[Patch(facecolor=S1, label="live element"),
                        Patch(facecolor="#f7cdb9", label="computed, but dead")],
               loc="lower center", ncol=2, frameon=False, fontsize=9,
               labelcolor=INK2, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Block granularity, drawn  -  N={N}, {block}x{block} tiles",
                 fontsize=12.5, color=INK, x=0.005, ha="left")
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    return fig


# ------------------------------------------------- experiment 1: granularity --
def fig_granularity(rows, n_show=16384):
    rows = sorted([r for r in rows if int(r["N"]) == n_show],
                  key=lambda r: r["waste_128x128"])
    names = [r["mask"] for r in rows]
    y = np.arange(len(rows))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 7.4), facecolor=SURF)
    _axis(axA); _axis(axB)

    h = 0.26
    for i, (key, col, lab) in enumerate([
            ("waste_128x128", S1, "128x128  (FlexAttention BlockMask)"),
            ("waste_32x32",   S2, "32x32  (ragged bounds)"),
            ("waste_16x16",   S3, "16x16  (MMA floor)")]):
        v = [r[key] for r in rows]
        off = (1 - i) * h
        axA.barh(y + off, v, height=h - 0.04, color=col, label=lab, zorder=3)
        if key != "waste_32x32":                      # selective labels
            for yy, vv in zip(y + off, v):
                axA.text(vv + 0.11, yy, f"{vv:.2f}", va="center", ha="left",
                         fontsize=7.5, color=INK2, zorder=4)
    axA.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    axA.text(1.0, len(rows) - 0.25, " no waste", fontsize=8, color=INK2, va="bottom")
    axA.set_yticks(y); axA.set_yticklabels(names, fontsize=9, color=INK)
    axA.set_xlim(0, 9.6)
    axA.set_xlabel("elements computed / live elements", fontsize=9.5, color=INK2)
    axA.set_title("A.  Finer tiles remove boundary waste, not strided waste",
                  fontsize=11.5, color=INK, loc="left", pad=12)
    axA.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)

    rem = np.array([r["waste_128x128"] - r["waste_16x16"] for r in rows])
    irr = np.array([r["waste_16x16"] - 1.0 for r in rows])
    GAP = 0.02
    axB.barh(y, rem, height=0.5, color=S1, zorder=3, label="removable by finer bounds")
    axB.barh(y, irr, left=rem + GAP, height=0.5, color=S2, zorder=3,
             label="irreducible at 16x16 (needs re-indexing)")
    for i, (a, b) in enumerate(zip(rem, irr)):
        if a > 0.40:
            axB.text(a / 2, i, f"{a:.2f}", va="center", ha="center",
                     fontsize=7.5, color="white", zorder=4)
        if b > 0.40:
            axB.text(a + GAP + b / 2, i, f"{b:.2f}", va="center", ha="center",
                     fontsize=7.5, color="white", zorder=4)
    axB.set_yticks(y); axB.set_yticklabels([]); axB.set_xlim(0, 8.4)
    axB.set_xlabel("excess elements computed, as a multiple of live",
                   fontsize=9.5, color=INK2)
    axB.set_title("B.  Where the waste actually lives", fontsize=11.5,
                  color=INK, loc="left", pad=12)
    axB.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.suptitle(f"Attention mask granularity study  -  sequence length {n_show:,}",
                 fontsize=13, color=INK, x=0.005, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return fig


# --------------------------------------------------- experiment 2: re-index --
def fig_reindex(sweep_rows, rx_rows, N=4096, order=None):
    from .experiments import reindex
    base = {r["mask"]: r for r in sweep_rows if int(r["N"]) == N}
    best = reindex.best_per_mask(rx_rows, (16, 16))
    order = order or list(best)
    rows = []
    for name in order:
        b = best[name]
        rows.append(dict(name=name, flex=base[name]["waste_128x128"],
                         ragged=base[name]["waste_16x16"], best=b["waste"],
                         tf=b["transform"], amort=b["amortisable"]))
    rows.sort(key=lambda r: r["flex"])
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor=SURF); _axis(ax)
    h = 0.26
    for i, (key, col, lab) in enumerate([
            ("flex",   S1, "BlockMask 128x128  (FlexAttention today)"),
            ("ragged", S2, "ragged affine bounds 16x16"),
            ("best",   S3, "+ best legal change of basis")]):
        ax.barh(y + (1 - i) * h, [r[key] for r in rows], height=h - 0.04,
                color=col, label=lab, zorder=3)
    for r, yy in zip(rows, y):
        ax.text(r["flex"] + 0.1, yy + h, f"{r['flex']:.2f}", va="center",
                fontsize=7.5, color=INK2)
        ax.text(r["best"] + 0.1, yy - h, f"{r['best']:.2f}", va="center",
                fontsize=7.5, color=INK2)
        if r["tf"] != "identity":
            ax.text(9.4, yy - h,
                    f"{r['tf']}  ({'free' if r['amort'] else 'costs traffic'})",
                    va="center", ha="right", fontsize=7.5, color=INK2, style="italic")
    ax.axvline(1.0, color=INK2, lw=1, ls=(0, (4, 3)), zorder=2)
    ax.set_yticks(y); ax.set_yticklabels([r["name"] for r in rows],
                                         fontsize=9, color=INK)
    ax.set_xlim(0, 9.6)
    ax.set_xlabel("elements computed / live elements", fontsize=9.5, color=INK2)
    ax.set_title("Two disjoint mechanisms, neither expressible in a boolean block mask",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.text(0.005, 0.955, f"Re-indexing study  -  sequence length {N:,}, exact",
             fontsize=13, color=INK, ha="left")
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


def write_all():
    """Regenerate every figure into docs/figures/ from the stored CSVs."""
    from .experiments import granularity, reindex
    sr, rr = granularity.read_csv(), reindex.read_csv()
    out = {"mask_gallery.png": fig_mask_gallery(),
           "granularity.png": fig_granularity(sr),
           "reindex.png": fig_reindex(sr, rr),
           "grain_dependence.png": fig_grain_dependence()}
    for name, fig in out.items():
        fig.savefig(FIGURES / name, dpi=170, facecolor=SURF)
    return list(out)


# ------------------------------------------ experiment 5: grain dependence ---
def fig_grain_dependence(mask_name="local256+str8", N=2048):
    """The argmin transform moves with max(BQ, A) -- so selection cannot be a
    per-mask lookup table. Evidence that a compiler is necessary.

    The x axis is max(BQ, A), not a mixed tile sequence: for a mask that depends
    only on (q - kv), waste is symmetric in the two tile dimensions and constant
    within each max class (experiment 6), so a single scalar names the column.
    """
    from . import masks as _m
    from .experiments import grain_dependence as gd
    m = {x.name: x for x in
         [_m.LocalStrided(256, 8), _m.LocalStrided(128, 4), _m.Dilated(8),
          _m.SinksWindow(4, 256)]}[mask_name]
    table, argmin = gd.sweep(m, N=N, grains=((128, 128), (64, 64), (32, 32), (16, 16)))
    grains = list(next(iter(table.values())))
    x = np.arange(len(grains))
    order = ["identity", "residue-perm-2", "residue-perm-4",
             "residue-perm-8", "residue-perm-16"]
    cols = {n: c for n, c in zip(order, (S1, S2, S3, S4, S5))}

    fig, ax = plt.subplots(figsize=(9.2, 5.4), facecolor=SURF)
    _axis(ax); ax.yaxis.grid(True, color=GRID, lw=1)
    winners = {argmin[g] for g in grains}
    ends = []
    for name in order:
        ys = [table[name][g] for g in grains]
        ax.plot(x, ys, lw=2, marker="o", ms=6, color=cols[name],
                label=name.replace("residue-perm-", "residue-perm "), zorder=3)
        if name in winners:                       # direct-label only the winners
            ends.append((ys[-1], name))
    span = max(max(v.values()) for v in table.values())
    ends.sort()
    for i, (y, name) in enumerate(ends):          # spread colliding end labels
        prev = ends[i - 1][0] if i else -1e9
        dy = 0.035 * span if (y - prev) < 0.05 * span else 0.0
        ax.annotate(name.replace("residue-perm-", "rp"), (x[-1], y),
                    xytext=(9, -6 if dy and i % 2 == 0 else 6 if dy else 0),
                    textcoords="offset points", va="center", fontsize=8.5,
                    color=cols[name], weight="bold")
    for i, g in enumerate(grains):                # ring the argmin at each grain
        best = argmin[g]
        ax.plot([i], [table[best][g]], marker="o", ms=13, mfc="none",
                mec=INK, mew=1.8, zorder=5)
    ax.set_xticks(x); ax.set_xticklabels([str(max(a, b)) for a, b in grains],
                                         fontsize=9.5, color=INK)
    ax.set_xlim(-0.3, len(grains) - 0.45)
    ax.set_xlabel("max(BQ, A)  -- the coarser tile dimension, a backend property",
                  fontsize=9.5, color=INK2)
    ax.set_ylabel("elements computed / live", fontsize=9.5, color=INK2)
    ax.set_title(f"The best transform is a function of max(BQ, A)  -  {mask_name}",
                 fontsize=12, color=INK, loc="left", pad=10)
    fig.text(0.005, 0.955, "Ringed = best at that tile shape."
             "  The winner moves rp2 -> rp4 -> rp8 as the coarser tile dimension"
             " falls: coarser tiles cannot absorb a deep fold's scatter.",
             fontsize=9, color=INK2, ha="left")
    ax.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


if __name__ == "__main__":
    matplotlib.use("Agg")
    print("wrote " + ", ".join(str(FIGURES / n) for n in write_all()))
