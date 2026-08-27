"""Direct tests for quantities that previously had only INDIRECT ones.

From an audit prompted by another session, which enumerated every quantity its
code computes and asked which had a direct test rather than being exercised
through something else. The one function with no direct test turned out to hold
three bugs; every other quantity was clean. That correlation is strong enough to
act on rather than argue with.

The three audited here, and why each was exposed:

  transforms.tile_stats   recently CHANGED (trailing-strip billing) and never
                          compared to an independent reference. A function you
                          just fixed is when its remaining defects are LEAST
                          likely to be found: the fix arrives with a passing
                          check attached, and it is the check you already had.
  cost.dense_cost         the reference cost.cost is validated against, and
                          itself never validated. A wrong oracle passes its own
                          tests.
  blockindex.build        feeds every gpu/ prediction, but its counts were
                          checked only against polyattn.cost -- another
                          implementation of the same idea -- and only inside a
                          GPU-gated test that has never run.

All three came back exact. The value is that they are now pinned.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

from polyattn import cost, masks, transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))


def naive_tile_stats(M, BQ, A):
    """Obviously correct: explicit loops, no reshape, no padding arithmetic."""
    nq, nk = M.shape
    tiles = elems = 0
    for i in range(0, nq, BQ):
        for j in range(0, nk, A):
            blk = M[i:i + BQ, j:j + A]
            if blk.any():
                tiles += 1
                elems += BQ * blk.shape[1]        # trailing strip at TRUE width
    return tiles, elems


@pytest.mark.parametrize("nk", [64, 100, 128, 187, 256])
@pytest.mark.parametrize("density", [0.02, 0.3, 0.9])
def test_tile_stats_matches_a_naive_reference(nk, density):
    """Non-multiple kv extents are the case the billing fix touched."""
    rng = np.random.default_rng(nk * 31 + int(density * 100))
    M = rng.random((128, nk)) < density
    for BQ in (16, 32, 64):
        for A in (16, 32, 64):
            assert transforms.tile_stats(M, BQ, A) == naive_tile_stats(M, BQ, A)


@pytest.mark.parametrize("m", [masks.Causal(), masks.SlidingWindow(64),
                               masks.Dilated(8), masks.DocPacked(64)],
                         ids=lambda m: m.name)
def test_dense_cost_matches_a_naive_reference(m):
    """cost.cost is validated against dense_cost; validate dense_cost too."""
    for N in (128, 256):
        M = np.stack([m.row_cols(q, N) for q in range(N)])
        for BQ, A in ((16, 16), (32, 64), (64, 32)):
            assert cost.dense_cost(m, N, BQ, A) == naive_tile_stats(M, BQ, A)[1]


@pytest.mark.parametrize("m", [masks.Causal(), masks.SlidingWindow(64),
                               masks.Dilated(8), masks.LocalStrided(64, 8),
                               masks.SinksWindow(4, 32)], ids=lambda m: m.name)
def test_blockindex_counts_and_index_are_correct(m):
    """Runs on CPU. blockindex only needs torch inside to_cuda()."""
    import blockindex
    for N in (128, 256):
        M = np.stack([m.row_cols(q, N) for q in range(N)])
        for BQ, A in ((16, 16), (32, 32), (64, 64), (128, 128)):
            if N % BQ or N % A:
                continue
            bi = blockindex.build(m, N, BQ, A)
            assert bi.elements == naive_tile_stats(M, BQ, A)[1]
            assert bi.live == int(M.sum())

            covered = np.zeros_like(M)
            for i in range(N // BQ):
                for jj in range(int(bi.kv_num[i])):
                    j = int(bi.kv_idx[i, jj])
                    blk = M[i * BQ:(i + 1) * BQ, j * A:(j + 1) * A]
                    assert blk.any(), "a listed tile is entirely dead"
                    covered[i * BQ:(i + 1) * BQ, j * A:(j + 1) * A] = True
                    assert (bi.kv_partial[i, jj] == 0) == bool(blk.all()), \
                        "full/partial flag disagrees with the tile contents"
            assert not (M & ~covered).any(), "a live element is in no listed tile"
