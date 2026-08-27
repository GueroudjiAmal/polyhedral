"""Brute-force oracle + evaluation. The oracle materialises; the selector must not."""
import sys, numpy as np
sys.path.insert(0, 'src'); sys.path.insert(0, 'scratch/2f')
from polyattn import transforms as T
from polyattn.transforms import tile_stats
from dset import DSet
import selector as S


def materialise(D, N):
    d = np.arange(N)[:, None] - np.arange(N)[None, :]
    M = np.zeros((N, N), bool)
    for lo, hi in D.ivs:
        M |= (d >= lo) & (d < hi)
    for r, s, lo, hi in D.aps:
        M |= (d >= lo) & (d < hi) & (d % s == r)
    return M


def oracle(D, N, BQ, A):
    M = materialise(D, N)
    out = {}
    fns = {"identity": T.t_identity, "shear": T.t_shear}
    for s in (2, 4, 8):
        fns[f"stridefold-{s}"] = T.make_stridefold(s)
    for s in (2, 3, 4, 6, 8, 12, 16, 32):
        fns[f"residue-perm-{s}"] = T.make_residue_perm(s)
    for nm, fn in fns.items():
        try:
            r = fn(M)
        except Exception:
            continue
        Mt = r[0] if isinstance(r, tuple) else r
        if Mt is None or Mt.sum() != M.sum():
            continue
        out[nm] = tile_stats(Mt, BQ, A)[1]
    return out
