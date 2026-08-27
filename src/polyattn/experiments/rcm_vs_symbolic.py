"""Experiment 4: symbolic re-indexing versus RCM, the published numerical baseline.

Binary Block Masking already reorders attention masks with RCM to raise block
density. So `residue-perm` is not competing with "no reordering" -- it is
competing with a heuristic that needs no knowledge of the predicate at all. The
question this settles: where does reading the permutation off the predicate beat
computing one from the matrix, and where does it not?
"""
import numpy as np

from .. import masks, reorder, transforms

N = 2048
GRAINS = ((128, 32), (16, 16))     # BBM's actual tile, and the aggressive floor


def default_zoo():
    return [masks.Dilated(2), masks.Dilated(4), masks.Dilated(8),
            masks.LocalStrided(256, 8), masks.SlidingWindow(128),
            masks.SinksWindow(4, 256), masks.DocPacked(512), masks.Causal()]


def best_symbolic(M, live, grain):
    """Best free (class A) transform derivable from the predicate."""
    best = (transforms.tile_stats(M, *grain)[1] / live, "identity")
    for s in (2, 4, 8, 16):
        Mt, _ = transforms.make_residue_perm(s)(M)
        w = transforms.tile_stats(Mt, *grain)[1] / live
        if w < best[0]:
            best = (w, f"residue-perm-{s}")
    return best


def run(zoo=None, N=N, grains=GRAINS, verbose=True):
    zoo = zoo or default_zoo()
    rcm = reorder.make_rcm()
    rows = []
    for m in zoo:
        M = m.dense(N)
        live = int(M.sum())
        Mr, _ = rcm(M)
        assert int(Mr.sum()) == live, "RCM lost elements"
        for grain in grains:
            base = transforms.tile_stats(M, *grain)[1] / live
            sym, sym_name = best_symbolic(M, live, grain)
            r = transforms.tile_stats(Mr, *grain)[1] / live
            rows.append(dict(mask=m.name, grain=f"{grain[0]}x{grain[1]}",
                             identity=base, symbolic=sym, symbolic_name=sym_name,
                             rcm=r))
    if verbose:
        print_table(rows)
    return rows


def print_table(rows):
    print(f"{'mask':<16}{'grain':>9}{'identity':>10}{'symbolic':>10}"
          f"{'RCM':>10}{'winner':>22}")
    print("-" * 77)
    last = None
    for r in rows:
        if last and last != r["mask"]:
            print()
        last = r["mask"]
        gap = r["rcm"] / r["symbolic"]
        if abs(gap - 1) < 0.02:
            win = "tie"
        elif gap > 1:
            win = f"symbolic {gap:.2f}x"
        else:
            win = f"RCM {1/gap:.2f}x"
        if r["symbolic_name"] != "identity":
            win += f"  [{r['symbolic_name'].replace('residue-perm-', 'rp')}]"
        print(f"{r['mask']:<16}{r['grain']:>9}{r['identity']:>10.3f}"
              f"{r['symbolic']:>10.3f}{r['rcm']:>10.3f}{win:>22}")
    print("\nidentity = no reordering | symbolic = best predicate-derived class A"
          "\nRCM      = reverse Cuthill-McKee on the materialised mask (Binary Block Masking's method)")


if __name__ == "__main__":
    print(f"N={N}\n")
    run()
