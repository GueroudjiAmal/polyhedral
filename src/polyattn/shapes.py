"""The shape library used by the composition search (experiment 3).

A *shape* is a structured sub-mask together with the free (class A) basis it
naturally wants. Decomposing a mask into shapes is how a predicate that is a
union of differently-structured families gets each family into its own basis.
"""
import numpy as np

from . import transforms

_DENSE_CACHE = {}


class Shape:
    def __init__(self, kind, p):
        self.kind, self.p = kind, p
        self.name = f"{kind}-{p}" if p is not None else kind

    def dense(self, N):
        key = (self.kind, self.p, N)
        if key not in _DENSE_CACHE:
            _DENSE_CACHE[key] = self._dense(N)
        return _DENSE_CACHE[key]

    def _dense(self, N):
        q = np.arange(N)[:, None]
        kv = np.arange(N)[None, :]
        causal = kv <= q
        if self.kind == "band":
            return causal & (kv > q - self.p)
        if self.kind == "lattice":
            return causal & ((q - kv) % self.p == 0)
        if self.kind == "prefix":
            return causal & (kv < self.p)
        if self.kind == "all":
            return causal
        raise ValueError(self.kind)

    def basis(self):
        """The free (class A) transform this shape wants. None means identity."""
        return transforms.make_residue_perm(self.p) if self.kind == "lattice" else None

    def __repr__(self):
        return f"Shape({self.name})"


LIBRARY = ([Shape("band", w) for w in (32, 64, 128, 256, 512)]
           + [Shape("lattice", s) for s in (2, 4, 8, 16)]
           + [Shape("prefix", g) for g in (4, 16)]
           + [Shape("all", None)])

#: canonical peel order -- bands first, so each element lands in the most
#: tile-dense home available before a sparser shape can claim it.
ORDER = {"band": 0, "lattice": 1, "prefix": 2, "all": 3}
