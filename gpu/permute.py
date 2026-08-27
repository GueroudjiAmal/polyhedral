"""Class A re-indexing on the GPU: permute K/V once, then every tile is contiguous.

The whole claim in docs/NOTES.md sec 4 is that this is FREE -- a q-independent
relabelling applied once per layer, O(N*d) traffic, after which tiles stay
rectangular. `permutation_cost_ms` measures that "once", separately, because
folding it into the kernel time would be exactly the flattery the log warns about.
"""
import numpy as np
import torch


def residue_perm(N, s):
    """Sort positions by (i mod s, i div s). Turns kv == q (mod s) block-diagonal."""
    i = np.arange(N)
    return np.argsort(i % s * N + i // s, kind="stable").astype(np.int32)


def identity_perm(N):
    return np.arange(N, dtype=np.int32)


def apply_perm(x, perm):
    """x: [BH, N, D] -> rows reordered so position p holds original perm[p]."""
    return x.index_select(1, perm).contiguous()


def invert(perm):
    inv = torch.empty_like(perm)
    inv[perm.long()] = torch.arange(perm.numel(), device=perm.device,
                                    dtype=perm.dtype)
    return inv


def permutation_cost_ms(q, k, v, perm, reps=50):
    """The one-time cost the 'free' claim depends on. Report it, never hide it."""
    from bench import time_ms
    return time_ms(lambda: (apply_perm(q, perm), apply_perm(k, perm),
                            apply_perm(v, perm)), reps=reps)[0]
