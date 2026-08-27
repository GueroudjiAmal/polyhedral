"""Experiment 2: which legal changes of basis remove the waste pruning cannot.

Hypothesis under test: strided waste is untouchable by finer bounds but removable
by re-indexing, because a strided live set is an affine lattice.

The legality argument, the class A / class B cost split, and the derivation of the
transform family all live in polyattn.transforms. This module runs the sweep.
"""
import csv

from .. import masks, transforms
from ..paths import RESULTS

N = 4096
GRAINS = [(32, 32), (16, 16)]


def default_zoo():
    return [masks.Causal(), masks.SlidingWindow(128), masks.SlidingWindow(256),
            masks.SlidingWindow(1024), masks.Dilated(2), masks.Dilated(4),
            masks.Dilated(8), masks.SinksWindow(4, 256),
            masks.LocalStrided(256, 8), masks.DocPacked(512)]


def run(N=N, zoo=None, grains=GRAINS, cands=None):
    zoo = zoo or default_zoo()
    cands = cands or transforms.candidates()
    rows = []
    for m in zoo:
        M0 = m.dense(N)
        live = int(M0.sum())
        base = {g: transforms.tile_stats(M0, *g)[1] for g in grains}
        for cname, fn in cands:
            Mt, meta = fn(M0)
            if Mt is None:
                continue
            assert int(Mt.sum()) == live, f"{m.name}/{cname} lost elements"
            kind, a, s = meta
            for BQ, A in grains:
                _, e = transforms.tile_stats(Mt, BQ, A)
                kvt = transforms.kv_per_tile(kind, a, s, BQ, A)
                rows.append(dict(mask=m.name, transform=cname, cls=kind,
                                 BQ=BQ, A=A, waste=e / live,
                                 flop_gain=base[(BQ, A)] / e,
                                 kv_rows_per_tile=kvt, traffic_ratio=kvt / A,
                                 amortisable=(kind == "A")))
    return rows


def best_per_mask(rows, grain=(16, 16)):
    """Lowest waste at `grain`, breaking ties in favour of the free (class A) transform."""
    out = {}
    for name in dict.fromkeys(r["mask"] for r in rows):
        sub = [r for r in rows if r["mask"] == name and (r["BQ"], r["A"]) == grain]
        out[name] = min(sub, key=lambda r: (round(r["waste"], 4), not r["amortisable"]))
    return out


def write_csv(rows, path=None):
    path = path or RESULTS / "reindex.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def read_csv(path=None):
    path = path or RESULTS / "reindex.csv"
    out = []
    for r in csv.DictReader(open(path)):
        r["BQ"] = int(r["BQ"]); r["A"] = int(r["A"])
        r["kv_rows_per_tile"] = int(r["kv_rows_per_tile"])
        for k in ("waste", "flop_gain", "traffic_ratio"):
            r[k] = float(r[k])
        r["amortisable"] = r["amortisable"] in (True, "True")
        out.append(r)
    return out


def print_table(rows, N_=None, grain=(16, 16)):
    print(f"N={N_ or N}, tile granularity {grain[0]}x{grain[1]}, exact (no sampling)\n")
    hdr = (f"{'mask':<16}{'best transform':<18}{'cls':>4}{'waste':>8}"
           f"{'vs id':>8}{'kv/tile':>9}{'amort?':>8}")
    print(hdr); print("-" * len(hdr))
    best = best_per_mask(rows, grain)
    for name, b in best.items():
        idw = [r for r in rows if r["mask"] == name and r["transform"] == "identity"
               and (r["BQ"], r["A"]) == grain][0]["waste"]
        print(f"{name:<16}{b['transform']:<18}{b['cls']:>4}{b['waste']:>8.2f}"
              f"{idw/b['waste']:>7.2f}x{b['kv_rows_per_tile']:>9}"
              f"{'yes' if b['amortisable'] else 'NO':>8}")
    print("\nwaste   = elements computed / live, after the transform (1.00 = perfect)")
    print("vs id   = FLOP reduction over the untransformed mask at the same granularity")
    print("kv/tile = distinct physical kv rows one tile must touch (A = contiguous)")
    print("amort?  = class A: permute K/V once per layer, then tiles are contiguous;")
    print("          class B: per-tile gather, cannot be amortised.")


if __name__ == "__main__":
    rows = run(); write_csv(rows); print_table(rows)
    print(f"\nwrote {RESULTS / 'reindex.csv'}")
