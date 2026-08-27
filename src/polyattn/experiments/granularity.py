"""Experiment 1: how much work does a boolean block mask waste?

One cost function expresses every model compared here -- see polyattn.cost. The
kill-threshold was stated in advance: if ragged 32-aligned bounds beat the
128x128 lattice by under ~15% across the zoo, the idea does not clear the noise
floor of kernel tuning.
"""
import csv

from .. import cost as model, masks
from ..paths import RESULTS

NS = [4096, 16384, 65536]
GRAINS = [(128, 128), (64, 64), (32, 32), (16, 16)]
BASE = (128, 128)     # FlexAttention's BlockMask granularity
FLOOR = (16, 16)      # MMA-shaped physical floor


def run(ns=NS, grains=GRAINS, zoo=None):
    zoo = zoo or masks.zoo()
    rows = []
    for m in zoo:
        for N in ns:
            live = m.live_count(N)
            if live == 0:
                continue
            rec = {"mask": m.name, "family": m.family,
                   "data_dependent": int(m.data_dependent), "N": N,
                   "density": live / (N * N), "live": live}
            sampled = False
            for BQ, A in grains:
                c, s = model.cost(m, N, BQ, A)
                sampled |= s
                rec[f"cost_{BQ}x{A}"] = c
                rec[f"waste_{BQ}x{A}"] = c / live
            rec["sampled"] = int(sampled)
            rec["speedup_vs_flex_32"] = rec[f"cost_{BASE[0]}x{BASE[1]}"] / rec["cost_32x32"]
            rec["speedup_vs_flex_16"] = (rec[f"cost_{BASE[0]}x{BASE[1]}"]
                                         / rec[f"cost_{FLOOR[0]}x{FLOOR[1]}"])
            rows.append(rec)
    return rows


def write_csv(rows, path=None):
    path = path or RESULTS / "results.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def read_csv(path=None):
    path = path or RESULTS / "results.csv"
    rows = []
    for r in csv.DictReader(open(path)):
        rows.append({k: (v if k in ("mask", "family") else float(v))
                     for k, v in r.items()})
    return rows


def print_table(rows, ns=None):
    hdr = (f"{'mask':<16}{'N':>7}{'density':>9}{'w@128':>8}{'w@64':>8}"
           f"{'w@32':>8}{'w@16':>8}{'gain32':>8}{'gain16':>8}")
    print(hdr); print("-" * len(hdr))
    last = None
    for r in rows:
        if ns and int(r["N"]) not in ns:
            continue
        if last and last != r["mask"]:
            print()
        last = r["mask"]
        print(f"{r['mask']:<16}{int(r['N']):>7}{r['density']*100:>8.2f}%"
              f"{r['waste_128x128']:>8.2f}{r['waste_64x64']:>8.2f}"
              f"{r['waste_32x32']:>8.2f}{r['waste_16x16']:>8.2f}"
              f"{r['speedup_vs_flex_32']:>8.2f}{r['speedup_vs_flex_16']:>8.2f}")


if __name__ == "__main__":
    rows = run(); write_csv(rows); print_table(rows)
    print(f"\nwrote {RESULTS / 'results.csv'}")


# ---------------------------------------------------------- the product grid --
# The square-grain sweep above conflates two independent knobs. Separating them
# changes the interpretation of experiment 1 -- see docs/NOTES.md §3a.
PRODUCT_BQS = (128, 64, 32, 16)
PRODUCT_AS = (128, 32, 16)


def product_grid(zoo=None, N=16384, bqs=PRODUCT_BQS, as_=PRODUCT_AS):
    """Waste over the full (BQ, A) product, not just the diagonal.

    Returns {mask_name: {(BQ, A): waste}}. The diagonal alone cannot tell a
    query-tile effect from a key-tile effect, and for attention they have very
    different costs: shrinking A is nearly free, shrinking BQ costs occupancy,
    MMA efficiency and per-tile softmax statistics.
    """
    from .. import cost as _cost
    zoo = zoo or [masks.SlidingWindow(128), masks.SlidingWindow(256),
                  masks.SlidingWindow(512), masks.SinksWindow(4, 256),
                  masks.DocPacked(512), masks.Dilated(8),
                  masks.LocalStrided(256, 8)]
    out = {}
    for m in zoo:
        live = m.live_count(N)
        out[m.name] = {(bq, a): _cost.cost(m, N, bq, a)[0] / live
                       for bq in bqs for a in as_}
    return out


def print_product_grid(grid, bqs=PRODUCT_BQS, as_=PRODUCT_AS):
    cols = [(bq, a) for bq in bqs for a in as_]
    print(f"{'mask':<16}" + "".join(f"{f'{bq}x{a}':>9}" for bq, a in cols))
    print("-" * (16 + 9 * len(cols)))
    for name, row in grid.items():
        print(f"{name:<16}" + "".join(f"{row[c]:>9.2f}" for c in cols))
    print("\nKV axis alone (BQ fixed at 128) vs query axis alone (A fixed at 128):")
    for name, row in grid.items():
        base = row[(128, 128)]
        print(f"  {name:<16} A 128->16: {base:.2f}->{row[(128, 16)]:.2f}"
              f" ({base/row[(128, 16)]:.2f}x)   "
              f"BQ 128->16: {base:.2f}->{row[(16, 128)]:.2f}"
              f" ({base/row[(16, 128)]:.2f}x)")
