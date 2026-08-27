"""EXPERIMENT 2 -- mechanism 1. docs/NOTES.md sec 4, and the go/no-go gate.

CLAIM UNDER TEST: a q-independent residue permutation collapses dilated-8's 8x
waste to ~1.03 at no traffic cost, because K/V is permuted ONCE per layer and
every tile stays rectangular and contiguous. Predicted element reduction 7.79x.

WHAT FALSIFIES IT: if the permuted kernel is not meaningfully faster than the
unpermuted one despite computing ~8x fewer elements, the transform buys nothing
in practice and mechanism 1 dies. The permutation cost is measured and reported
SEPARATELY -- folding it in would flatter the method, and amortising it over
layers is a modelling choice the reader should get to make.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute               # noqa: E402
from triton_attn import block_sparse_attention             # noqa: E402

N, D, BH = 4096, 64, 8
CASES = [("dilated-4", 4), ("dilated-8", 8), ("local256+str8", 8)]
TILES = [(128, 128), (128, 32), (64, 64), (32, 32), (16, 16)]


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    for name, s in CASES:
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        M = np.stack([m.row_cols(i, N) for i in range(N)])
        p_np = permute.residue_perm(N, s)
        perm = torch.from_numpy(p_np).cuda()
        Mp = M[p_np][:, p_np]
        qp, kp, vp = (permute.apply_perm(x, perm) for x in (q, k, v))
        pcost_ms = permute.permutation_cost_ms(q, k, v, perm)

        rows = []
        for BQ, A in TILES:
            bi0 = blockindex.build(_Dense(M), N, BQ, A)
            bi1 = blockindex.build(_Dense(Mp), N, BQ, A)
            i0, i1 = blockindex.to_cuda(bi0), blockindex.to_cuda(bi1)
            t0 = bench.time_ms(lambda: block_sparse_attention(
                q, k, v, *i0, kind, p0, p1, p2, BQ, A))[0]
            t1 = bench.time_ms(lambda: block_sparse_attention(
                qp, kp, vp, *i1, kind, p0, p1, p2, BQ, A,
                perm_q=perm.int(), perm_kv=perm.int()))[0]
            rows.append([f"{BQ}x{A}", bi0.waste, bi1.waste,
                         bi0.elements / bi1.elements, t0, t1, t0 / t1])
        bench.report(
            rows,
            [("tile", 10), ("waste id", 10), ("waste perm", 12), ("elem gain", 11),
             ("ms id", 9), ("ms perm", 10), ("time gain", 11)],
            title=f"{name}  residue-perm-{s}   N={N} BH={BH}",
            note=f"one-time permutation of Q/K/V: {pcost_ms:.3f} ms "
                 f"(NOT included in 'ms perm'; amortised over layers in practice)\n"
                 "elem gain is the prediction, time gain is the measurement.")


if __name__ == "__main__":
    main()
