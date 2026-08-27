"""Experiment 6: what variable does transform selection actually depend on?

§5b claimed "finer tiles favour deeper folds", but stated it over the sequence
128x128, 128x32, 64x16, 32x32, 16x16 -- which moves BOTH axes at once. That is
the same confound §3a caught in the granularity model, recurring one level up.

Tested here on the full (BQ, A) product:

  * SYMMETRY   is waste(BQ, A) == waste(A, BQ)?
  * MAX-LAW    is waste constant across all cells sharing max(BQ, A)?

If both hold, selection for that mask is a function of a single scalar -- the
COARSER tile dimension -- which is a cost model with structure rather than a
lookup table with a caveat.
"""
import numpy as np

from .. import masks, reorder, transforms

N = 2048
DIMS = (128, 64, 32, 16)
PERMS = (2, 4, 8, 16)


def default_zoo():
    return [masks.LocalStrided(256, 8), masks.Dilated(8), masks.Dilated(4),
            masks.SlidingWindow(128), masks.SinksWindow(4, 256),
            masks.DocPacked(512)]


def grid(m, N=N, dims=DIMS, perms=PERMS):
    """{(BQ, A): (best waste, best transform name)} over the full product."""
    M = m.dense(N)
    live = int(M.sum())
    variants = {"identity": M}
    for s in perms:
        variants[f"rp{s}"] = transforms.make_residue_perm(s)(M)[0]
    out = {}
    for bq in dims:
        for a in dims:
            best = min(((transforms.tile_stats(Mv, bq, a)[1] / live, n)
                        for n, Mv in variants.items()), key=lambda t: t[0])
            out[(bq, a)] = best
    return out


def symmetry(g, dims=DIMS):
    """Max |waste(BQ,A) - waste(A,BQ)| over off-diagonal pairs."""
    devs = [abs(g[(a, b)][0] - g[(b, a)][0])
            for i, a in enumerate(dims) for b in dims[i + 1:]]
    return max(devs) if devs else 0.0


def max_law(g, dims=DIMS):
    """Max spread of waste within each max(BQ,A) class; 0 means the law holds."""
    by = {}
    for (bq, a), (w, _) in g.items():
        by.setdefault(max(bq, a), []).append(w)
    return {k: (max(v) - min(v)) for k, v in sorted(by.items(), reverse=True)}


def run(zoo=None, N=N, dims=DIMS, verbose=True):
    zoo = zoo or default_zoo()
    out = {}
    for m in zoo:
        g = grid(m, N, dims)
        sym, spread = symmetry(g, dims), max_law(g, dims)
        out[m.name] = dict(grid=g, symmetry=sym, max_spread=spread)
        if not verbose:
            continue
        print(f"\n{m.name}   N={N}   argmin transform (waste) per (BQ, A)")
        print(f"{'BQ\\A':>6}" + "".join(f"{a:>14}" for a in dims))
        for bq in dims:
            cells = "".join(f"{g[(bq,a)][1]}({g[(bq,a)][0]:.2f})".rjust(14)
                            for a in dims)
            print(f"{bq:>6}{cells}")
        worst = max(spread.values())
        print(f"  symmetry: max |w(BQ,A)-w(A,BQ)| = {sym:.3f}"
              f"   -> {'SYMMETRIC' if sym < 0.01 else 'ASYMMETRIC'}")
        print(f"  max-law : spread within each max(BQ,A) class = "
              + ", ".join(f"{k}:{v:.3f}" for k, v in spread.items())
              + f"   -> {'HOLDS' if worst < 0.01 else 'FAILS'}")
    return out


def rcm_damage_by_max(m, N=N, dims=DIMS):
    """RCM damage re-indexed by max(BQ,A) -- does the drift straighten?"""
    M = m.dense(N)
    live = int(M.sum())
    Mr, _ = reorder.make_rcm()(M)
    by = {}
    for bq in dims:
        for a in dims:
            r = (transforms.tile_stats(Mr, bq, a)[1]
                 / transforms.tile_stats(M, bq, a)[1])
            by.setdefault(max(bq, a), []).append(r)
    return {k: (float(np.mean(v)), float(max(v) - min(v)))
            for k, v in sorted(by.items(), reverse=True)}


if __name__ == "__main__":
    res = run()
    print("\n\nRCM damage indexed by max(BQ, A)   [mean, spread within class]")
    for m in (masks.LocalStrided(256, 8), masks.SinksWindow(4, 256)):
        d = rcm_damage_by_max(m)
        print(f"  {m.name:<16}" + "  ".join(
            f"max={k}: {v[0]:.3f} (+-{v[1]:.3f})" for k, v in d.items()))
