"""Brute-force oracle. Materialises the mask, applies the transform, counts tiles.
Written from scratch here rather than imported, so a bug in polyattn cannot make
the selector look right. Bills a trailing partial column strip at its TRUE width,
matching the fix d4 applied -- stated explicitly because otherwise a harness
convention gets mistaken for a result."""
import numpy as np
from blocks import Iv, Ap


def dense(pieces, N):
    d = np.arange(N)[:, None] - np.arange(N)[None, :]
    M = np.zeros((N, N), bool)
    for p in pieces:
        if isinstance(p, Iv):
            M |= (d >= p.lo) & (d < p.hi)
        elif p.count > 0:
            last = p.start + (p.count - 1) * p.stride
            M |= (d >= p.start) & (d <= last) & ((d - p.start) % p.stride == 0)
    return M


def tiles_cost(M, BQ, A):
    nq, nk = M.shape
    pq, pk = (-nq) % BQ, (-nk) % A
    P = np.pad(M, ((0, pq), (0, pk)))
    t = P.reshape(-1, BQ, P.shape[1] // A, A).any(axis=(1, 3))
    n = int(t.sum())
    if pk:                                   # trailing strip billed at true width
        last = int(t[:, -1].sum())
        return (n - last) * BQ * A + last * BQ * (A - pk)
    return n * BQ * A


def apply_xform(M, name):
    n = M.shape[0]
    if name == "identity":
        return M
    if name == "shear":
        q, kv = np.nonzero(M)
        if not len(q):
            return None
        j = kv - q; off = j.min()
        out = np.zeros((n, int(j.max() - off + 1)), bool); out[q, j - off] = True
        return out
    if name.startswith("stridefold-"):
        s = int(name.split("-")[1])
        q, kv = np.nonzero(M)
        d = q - kv
        if not len(q) or not ((d % s == 0) & (d >= 0)).all():
            return None
        j = d // s
        out = np.zeros((n, int(j.max() + 1)), bool); out[q, j] = True
        return out
    if name.startswith("residue-perm-"):
        s = int(name.split("-")[-1])
        if n % s:
            return None
        order = np.argsort(np.arange(n) % s * n + np.arange(n) // s, kind="stable")
        return M[order][:, order]
    raise KeyError(name)


def oracle_cost(pieces, N, BQ, A, name):
    M = apply_xform(dense(pieces, N), name)
    return None if M is None else tiles_cost(M, BQ, A)


def oracle_select(pieces, N, BQ, A, candidates):
    M0 = dense(pieces, N)
    best = None
    for nm in candidates:
        M = apply_xform(M0, nm)
        if M is None:
            continue
        c = tiles_cost(M, BQ, A)
        key = (c, 0 if (nm == "identity" or nm.startswith("residue")) else 1, nm)
        if best is None or key < best[0]:
            best = (key, nm, c)
    return (best[1], best[2]) if best else (None, None)
