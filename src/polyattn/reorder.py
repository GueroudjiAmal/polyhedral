"""Reverse Cuthill-McKee -- the numerical baseline for class A re-indexing.

Binary Block Masking (arXiv:2409.15097, Sept 2024) applies RCM to the attention
mask matrix before the kernel, explicitly to concentrate scattered non-zeros so
that fewer tiles are occupied. That is class A re-indexing, published, and it is
the incumbent this project's symbolic transforms have to beat or explain.

The two methods differ in kind, which is the whole question:

  RCM            a graph-bandwidth heuristic run on a MATERIALISED N x N mask.
                 O(N^2) to build, no guarantee, but needs no knowledge of where
                 the mask came from -- it works on anything.
  residue-perm   a closed-form permutation read off the PREDICATE. Exact by
                 construction, nothing materialised, but only fires when the
                 predicate has the lattice structure it looks for.

Implemented here rather than pulled from scipy so the comparison has no hidden
dependency, and with the George-Liu pseudo-peripheral start so the baseline is
the real algorithm and not a strawman.
"""
import numpy as np


def _bfs_levels(A, root):
    """Level structure of the component containing `root`."""
    n = A.shape[0]
    seen = np.zeros(n, bool)
    seen[root] = True
    levels, frontier = [np.array([root])], np.array([root])
    while True:
        nxt = np.flatnonzero(A[frontier].any(axis=0) & ~seen)
        if nxt.size == 0:
            return levels, seen
        seen[nxt] = True
        levels.append(nxt)
        frontier = nxt


def _pseudo_peripheral(A, start, deg):
    """George-Liu: walk to a node of near-maximal eccentricity."""
    u = start
    levels, _ = _bfs_levels(A, u)
    for _ in range(10):
        last = levels[-1]
        v = last[np.argmin(deg[last])]
        new, _ = _bfs_levels(A, v)
        if len(new) <= len(levels):
            return u
        u, levels = v, new
    return u


def rcm_order(A):
    """Reverse Cuthill-McKee ordering of a symmetric boolean adjacency matrix.

    Components are handled in turn, so a mask that decomposes into independent
    pieces gets those pieces laid out contiguously -- which matters, because it
    is how RCM can stumble onto a lattice structure without being told about it.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    visited = np.zeros(n, bool)
    order = []
    while not visited.all():
        cand = np.where(visited, n + 1, deg)
        root = _pseudo_peripheral(A, int(np.argmin(cand)), deg)
        queue, head = [root], 0
        visited[root] = True
        while head < len(queue):
            v = queue[head]; head += 1
            order.append(v)
            nbr = np.flatnonzero(A[v] & ~visited)
            if nbr.size:
                nbr = nbr[np.argsort(deg[nbr], kind="stable")]
                visited[nbr] = True
                queue.extend(int(x) for x in nbr)
    return np.asarray(order, dtype=np.int64)[::-1]


def make_rcm(symmetrise=True):
    """RCM as a transform, matching the signature of polyattn.transforms.

    Attention masks are not symmetric (causal), so the graph is symmetrised
    before ordering; the resulting permutation is applied to BOTH axes, which is
    what RCM means and is legal here -- permuting queries is free and permuting
    keys is class A.
    """
    def f(M):
        A = (M | M.T) if symmetrise else M
        np.fill_diagonal(A, False)
        p = rcm_order(A)
        return M[p][:, p], ("A", 0, 1)
    f.__name__ = "rcm"
    return f
