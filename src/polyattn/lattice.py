"""Deriving the transform from the predicate's lattice, instead of guessing it.

THIS IS THE PIECE THE PROJECT NAME PROMISES AND THE CODE HAS BEEN FAKING.

`polyattn.selector` picks a transform by scoring a hardcoded list of thirteen
candidates -- residue-perm-{2,3,4,6,8,12,16,32} and friends -- and returning the
best. That is enumeration, not derivation. It answers "which of eight guesses
scored best", not "what transform does this predicate call for", and the
difference shows up in two ways this module makes visible:

  * where the lattice is clean, the answer is COMPUTABLE in O(#runs) and the
    candidate list is redundant;
  * where the derived answer is not IN the list, or not expressible at this N,
    enumeration silently returns `identity` and reports agreement -- concealing
    that the right transform exists and could not be applied.

The offset set D of a diagonally-invariant mask generates a sublattice of Z. For
a rank-1 lattice its generator is the fold depth that makes the mask
block-diagonal -- the Hermite-normal-form step, specialised to one dimension.
Where D contains runs wider than a point the generated lattice is Z, no single
fold applies, and the honest answer is that the mask needs DECOMPOSING rather
than folding (NOTES §5), which is the composition result arriving from the
lattice side rather than from a search.
"""
from functools import reduce
from math import gcd

__all__ = ["derive_fold", "derive_transform", "LatticeResult"]


class LatticeResult:
    """What the predicate's lattice says, and whether it can be acted on."""

    def __init__(self, fold, reason, expressible, name):
        self.fold, self.reason = fold, reason
        self.expressible, self.name = expressible, name

    def __repr__(self):
        return (f"LatticeResult(fold={self.fold}, name={self.name!r}, "
                f"expressible={self.expressible}, reason={self.reason!r})")


def derive_fold(runs):
    """Generator of the sublattice of Z spanned by D. 1 means no single fold.

    O(#runs), no matrix, no candidate list, no search.
    """
    if not runs:
        return 1
    if any(r - l + 1 > 1 for l, r in runs):
        return 1                      # a run wider than a point spans Z
    pts = sorted(l for l, _ in runs)
    return reduce(gcd, (b - a for a, b in zip(pts, pts[1:])), 0) or 1


def derive_transform(runs, N, candidates=None):
    """Derive the transform, and say plainly when it cannot be applied.

    The point of returning `expressible=False` rather than falling back to
    identity: a selector that silently substitutes identity reports agreement
    with a cost model that also cannot see the missing transform, so the failure
    is invisible from both sides. `dilated-3` at N=2048 is exactly this -- the
    derived fold is 3, `residue-perm-3` needs N % 3 == 0, and the enumerating
    selector answers `identity` with no indication that the right answer existed.
    """
    f = derive_fold(runs)
    if f <= 1:
        return LatticeResult(1, "lattice is Z -- no single fold; decompose (NOTES §5)",
                             True, "identity")
    name = f"residue-perm-{f}"
    if N % f:
        return LatticeResult(f, f"N={N} not divisible by the derived fold {f}",
                             False, name)
    if candidates is not None and name not in candidates:
        return LatticeResult(f, f"derived fold {f} is not in the candidate list",
                             False, name)
    return LatticeResult(f, "derived from the offset lattice", True, name)
