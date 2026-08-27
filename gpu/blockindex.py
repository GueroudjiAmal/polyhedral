"""Build block-sparse tile indices from a polyattn mask, at any (BQ, A).

This is the bridge: the GPU kernels are driven by exactly the same tiling the
analytical cost model counts, so a wall-clock number and a predicted element
count refer to the same set of tiles. If they diverge, the cost model is wrong --
which is the entire point of running any of this.

Mirrors FlexAttention's BlockMask distinction: a tile is FULL (every element
live, so the mask can be skipped inside it) or PARTIAL (computed, then masked).
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BlockIndex:
    BQ: int
    A: int
    N: int
    kv_idx: np.ndarray       # [n_q_blocks, max_kv] int32 -- kv block ids to visit
    kv_num: np.ndarray       # [n_q_blocks] int32 -- how many are valid
    kv_partial: np.ndarray   # [n_q_blocks, max_kv] int8 -- 1 if mask must be applied
    elements: int            # tiles * BQ * A -- the cost model's prediction
    live: int

    @property
    def waste(self):
        return self.elements / self.live


def build(mask, N, BQ, A):
    """Offline, like create_block_mask. Cost is not part of any measurement."""
    assert N % BQ == 0 and N % A == 0
    M = np.stack([mask.row_cols(q, N) for q in range(N)])
    live = int(M.sum())
    tiles = M.reshape(N // BQ, BQ, N // A, A)
    any_live = tiles.any(axis=(1, 3))
    all_live = tiles.all(axis=(1, 3))

    nq = N // BQ
    counts = any_live.sum(axis=1)
    max_kv = int(counts.max())
    kv_idx = np.zeros((nq, max_kv), dtype=np.int32)
    kv_partial = np.zeros((nq, max_kv), dtype=np.int8)
    for i in range(nq):
        cols = np.flatnonzero(any_live[i])
        kv_idx[i, :cols.size] = cols
        kv_partial[i, :cols.size] = (~all_live[i, cols]).astype(np.int8)
    return BlockIndex(BQ, A, N, kv_idx, counts.astype(np.int32), kv_partial,
                      int(any_live.sum()) * BQ * A, live)


def to_cuda(bi, device="cuda"):
    import torch
    return (torch.from_numpy(bi.kv_idx).to(device),
            torch.from_numpy(bi.kv_num).to(device),
            torch.from_numpy(bi.kv_partial).to(device))
