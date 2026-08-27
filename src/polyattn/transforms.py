"""The legal change-of-basis library, and the tile metrics used to score it.

LEGALITY CONSTRAINT (this is what bounds the search space):
attention reduces over kv for a fixed q -- out[q] = sum_kv softmax(S[q,kv]) V[kv].
So a transform may NOT mix q into the reduced axis in a way that breaks the
fibration over q.  The legal family is therefore

    q'  = pi_q(q)                       (any permutation of queries -- free, undo on output)
    kv' = pi_kv(kv)                     CLASS A: q-independent relabelling of keys
    kv' = (kv - a*q) / s                CLASS B: q-dependent shear / stride-fold

and the two classes have completely different costs:

  CLASS A  the permuted K/V tensor is materialised ONCE per layer, O(N*d) traffic,
           after which every tile is rectangular and contiguous.  Essentially free.
  CLASS B  every tile needs a different slice of K/V, so the gather cannot be
           amortised: a BQ x A tile touches A + a*(BQ-1) distinct kv rows instead
           of A.  FLOPs go down, traffic goes up.

Everything here is computed on a materialised N x N mask -- exact, no closed forms,
no sampling.  N is kept small enough for that to be cheap.
"""
import numpy as np
from . import masks



# ---------------------------------------------------------------- transforms --
def t_identity(M):
    return M, ("A", 0, 1)


def t_shear(M):
    """CLASS B: kv' = kv - q.  Straightens any diagonal band into a rectangle."""
    n = M.shape[0]
    q, kv = np.nonzero(M)
    j = kv - q
    off = j.min()
    W = int(j.max() - off + 1)
    out = np.zeros((n, W), dtype=bool)
    out[q, j - off] = True
    return out, ("B", 1, 1)          # a=1, s=1


def make_stridefold(s):
    def f(M):
        """CLASS B: kv' = (q - kv)/s.  Folds a stride-s lattice onto consecutive ints."""
        n = M.shape[0]
        q, kv = np.nonzero(M)
        d = q - kv
        keep = (d % s == 0) & (d >= 0)
        if not keep.all():                     # mask has off-lattice elements too
            return None, None
        j = d[keep] // s
        W = int(j.max() + 1)
        out = np.zeros((n, W), dtype=bool)
        out[q[keep], j] = True
        return out, ("B", 1, s)
    f.__name__ = f"stridefold-{s}"
    return f


def make_residue_perm(s):
    def f(M):
        """CLASS A: sort BOTH axes by (i mod s, i div s).

        A mask defined by kv == q (mod s) becomes block-diagonal: s independent
        dense sub-problems.  This is what LongNet does by hand, as a transform.
        """
        n = M.shape[0]
        order = np.argsort(np.arange(n) % s * n + np.arange(n) // s, kind="stable")
        return M[order][:, order], ("A", 0, 1)
    f.__name__ = f"residue-perm-{s}"
    return f


# ------------------------------------------------------------------- metrics --
def tile_stats(M, BQ, A):
    """(tiles computed, elements computed).

    Class B transforms return NON-SQUARE matrices whose kv extent need not be a
    multiple of A. Zero-padding cannot invent a tile -- an all-zero tile is never
    counted -- but billing every tile a full BQ*A over-charges the final partial
    column strip, which inflated every shear and stridefold waste number. The
    trailing strip is now billed at its real width.
    """
    nq, nk = M.shape
    pq, pk = (-nq) % BQ, (-nk) % A
    if pq or pk:
        M = np.pad(M, ((0, pq), (0, pk)))
    t = M.reshape(-1, BQ, M.shape[1] // A, A).any(axis=(1, 3))
    n = int(t.sum())
    if pk:                      # last column-block holds only (A - pk) real columns
        last = int(t[:, -1].sum())
        return n, (n - last) * BQ * A + last * BQ * (A - pk)
    return n, n * BQ * A


def kv_per_tile(kind, a, s, BQ, A):
    """Distinct physical kv rows a single BQ x A tile must touch.

    Class A: the permutation is applied to K/V once, so a tile reads A contiguous
    rows of the permuted tensor.  Class B: physical kv = j*s + a*q + const, so the
    tile spans the offset set {a*dq + s*dj}.
    """
    if kind == "A":
        return A
    dq = np.arange(BQ)[:, None]
    dj = np.arange(A)[None, :]
    return int(np.unique(a * dq - s * dj).size)



def candidates():
    """Every transform the search may use, in a stable order."""
    c = [("identity", t_identity), ("shear", t_shear)]
    for s in (2, 4, 8):
        c.append((f"residue-perm-{s}", make_residue_perm(s)))
        c.append((f"stridefold-{s}", make_stridefold(s)))
    return c
