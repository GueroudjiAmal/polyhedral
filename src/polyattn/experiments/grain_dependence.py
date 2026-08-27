"""Experiment 5: the best transform depends on the TILE SHAPE, not just the mask.

Spotted in experiment 4's own output and nearly missed. For `local256+str8`:

    128x32   best is residue-perm-2   (waste 2.411)
    16x16    best is residue-perm-8   (waste 1.332)

The winning transform *changes*, not merely the winning margin. Folding by 8
scatters the local band across a 32-wide column tile badly enough that a milder
fold wins instead.

WHY THIS IS THE LOAD-BEARING RESULT. Selection is not a function of the predicate.
It is a function of (predicate, tile shape) -- and tile shape is a backend and
hardware property, not a property of the mask. So:

  * no per-pattern hand implementation can do it -- LongNet and Sparse Transformer
    bake one basis in at authoring time and cannot re-derive it when the tile
    shape or the hardware changes;
  * RCM cannot do it either -- it minimises graph bandwidth, an objective with no
    tile-shape input at all (see rcm_grain_invariance below);
  * a static per-family lookup table cannot do it, which is precisely what makes
    a compiler necessary rather than decorative.

A regression argues selection is SAFER. This argues selection is NECESSARY.
"""
import numpy as np

from .. import masks, reorder, transforms

N = 2048
GRAINS = ((128, 128), (128, 32), (64, 16), (32, 32), (16, 16))
PERMS = (2, 4, 8, 16)


def default_zoo():
    return [masks.LocalStrided(256, 8), masks.LocalStrided(128, 4),
            masks.Dilated(8), masks.SinksWindow(4, 256)]


def sweep(m, N=N, grains=GRAINS, perms=PERMS):
    """{transform_name: {grain: waste}} plus the argmin transform per grain."""
    M = m.dense(N)
    live = int(M.sum())
    variants = {"identity": M}
    for s in perms:
        variants[f"residue-perm-{s}"] = transforms.make_residue_perm(s)(M)[0]
    table = {name: {g: transforms.tile_stats(Mv, *g)[1] / live for g in grains}
             for name, Mv in variants.items()}
    argmin = {g: min(table, key=lambda n: table[n][g]) for g in grains}
    return table, argmin


def rcm_grain_invariance(m, N=N, grains=GRAINS):
    """Does RCM's damage depend on the tile shape? If not, it is optimising a
    tile-agnostic objective -- which would explain the regressions rather than
    merely record them."""
    M = m.dense(N)
    live = int(M.sum())
    Mr, _ = reorder.make_rcm()(M)
    return {g: (transforms.tile_stats(Mr, *g)[1] / live)
               / (transforms.tile_stats(M, *g)[1] / live) for g in grains}


def run(zoo=None, N=N, grains=GRAINS, verbose=True):
    zoo = zoo or default_zoo()
    out = {}
    for m in zoo:
        table, argmin = sweep(m, N, grains)
        out[m.name] = dict(table=table, argmin=argmin)
        if not verbose:
            continue
        print(f"\n{m.name}   N={N}")
        print(f"{'transform':<18}" + "".join(f"{f'{a}x{b}':>10}" for a, b in grains))
        print("-" * (18 + 10 * len(grains)))
        for name, row in table.items():
            marks = "".join(
                f"{row[g]:>9.3f}" + ("*" if argmin[g] == name else " ") for g in grains)
            print(f"{name:<18}{marks}")
        winners = {argmin[g] for g in grains}
        print(f"  argmin per grain: " + ", ".join(
            f"{a}x{b} -> {argmin[(a,b)]}" for a, b in grains))
        print(f"  --> {'SELECTION IS GRAIN-DEPENDENT' if len(winners) > 1 else 'one transform wins everywhere'}")

    if verbose:
        print("\n\nRCM damage ratio (RCM waste / identity waste), by grain:")
        print(f"{'mask':<18}" + "".join(f"{f'{a}x{b}':>10}" for a, b in grains))
        print("-" * (18 + 10 * len(grains)))
        for m in zoo:
            r = rcm_grain_invariance(m, N, grains)
            print(f"{m.name:<18}" + "".join(f"{r[g]:>10.3f}" for g in grains))
        print("\nA flat row means RCM's objective has no tile-shape input:"
              " it is optimising the wrong thing, not optimising badly.")
    return out


if __name__ == "__main__":
    run()
