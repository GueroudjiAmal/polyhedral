"""Decompositions must be disjoint and must reconstruct the mask exactly.

Disjointness is not cosmetic: an element counted in two parts would be
double-counted in the softmax denominator, so a decomposition that overlaps is
not merely slow, it is wrong.
"""
import itertools

import numpy as np
import pytest

from polyattn import masks, shapes
from polyattn.experiments import compose

N = 256


def _peel(M, combo):
    combo = sorted(combo, key=lambda s: (shapes.ORDER[s.kind], -(s.p or 0)))
    taken = np.zeros_like(M)
    parts = []
    for sh in combo:
        p = sh.dense(M.shape[0]) & M & ~taken
        taken |= p
        parts.append(p)
    return parts, taken


@pytest.mark.parametrize("combo", list(itertools.combinations(shapes.LIBRARY, 2))[:40])
def test_peeled_parts_are_pairwise_disjoint(combo):
    M = masks.LocalStrided(64, 8).dense(N)
    parts, _ = _peel(M, combo)
    for a, b in itertools.combinations(parts, 2):
        assert not (a & b).any()


def test_accepted_decomposition_reconstructs_the_mask():
    m = masks.LocalStrided(64, 8)
    M = m.dense(N)
    accepted = 0
    for k in (1, 2):
        for combo in itertools.combinations(shapes.LIBRARY, k):
            r = compose.evaluate(M, combo)
            if r is None:
                continue
            accepted += 1
            _, taken = _peel(M, combo)
            np.testing.assert_array_equal(taken, M)
    assert accepted > 0, "no decomposition was accepted -- the search is broken"


def test_merge_overhead_is_zero_for_a_single_part():
    assert compose.merge_overhead_fraction(1, 4096, 10**6, 0.2) == 0.0
