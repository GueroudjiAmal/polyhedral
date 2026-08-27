"""The granularity cost model (polyattn.cost).

cost(BQ, A) = for each query row-block of BQ rows, take the union of live kv
columns over the rows in that block, cover it with A-aligned column segments,
and charge BQ * A for every covered segment.

This one function expresses every model we care about:

  cost(128, 128)  FlexAttention's BlockMask: a (row-block, col-block) tile is
                  computed in full if any element in it is live.  Partial blocks
                  cost the same as full blocks.
  cost(BQ, A)     polyhedral ragged bounds at tile granularity (BQ, A).
  cost(1, 1)      the live element count -- unreachable, the true lower bound.

A=16 is the MMA-shaped floor: tensor cores cannot skip work below it, so
cost(16, 16) is the physically achievable optimum, not cost(1, 1).
"""
import numpy as np

MAX_BLOCKS = 512          # above this we sample row-blocks evenly and scale


def cost(mask, N, BQ, A, exact_only=False):
    """Returns (elements_computed, was_sampled)."""
    assert N % BQ == 0 and N % A == 0
    nblocks = N // BQ
    if nblocks <= MAX_BLOCKS or exact_only:
        idx = np.arange(nblocks)
        sampled = False
    else:
        idx = np.unique(np.linspace(0, nblocks - 1, MAX_BLOCKS).round().astype(int))
        sampled = True

    acc = 0
    for b in idx:
        q0 = int(b) * BQ
        u = mask.union_cols(q0, q0 + BQ, N)
        ncov = int(u.reshape(-1, A).any(axis=1).sum())
        acc += BQ * A * ncov
    if sampled:
        acc = acc * nblocks / len(idx)
    return float(acc), sampled


def dense_cost(mask, N, BQ, A):
    """Brute-force reference over a materialised N x N mask. Small N only."""
    M = mask.dense(N)
    tiles = M.reshape(N // BQ, BQ, N // A, A).any(axis=(1, 3))
    return float(tiles.sum() * BQ * A)
