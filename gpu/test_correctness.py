"""Run this BEFORE any experiment. Timings on a wrong kernel are worthless.

Checks, in order:
  1. the numpy mask and the torch mask agree (three consumers, one definition)
  2. BlockIndex's element count equals polyattn.cost's prediction
  3. the Triton kernel matches dense reference attention at every tile shape
  4. the class A permuted path matches the same reference
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, masks_gpu, permute, reference          # noqa: E402
from triton_attn import block_sparse_attention            # noqa: E402

from polyattn import cost as pcost                        # noqa: E402

N, D, BH = 1024, 64, 2
TILES = [(128, 128), (128, 32), (64, 64), (32, 32), (16, 16), (128, 16), (16, 128)]
NAMES = ["causal", "window-128", "dilated-8", "local256+str8", "twoband-1000"]
TOL = 3e-3           # fp16 accumulation in fp32; reference is fp32


def main():
    torch.manual_seed(0)
    fails = []

    print("=== 1. mask definitions agree across consumers ===")
    for n in masks_gpu.SPECS:
        ok = masks_gpu.check_agreement(n, N=512)
        print(f"  {n:<16} {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append(f"mask disagreement: {n}")

    print("\n=== 2. BlockIndex element count == polyattn.cost prediction ===")
    for n in NAMES:
        m = masks_gpu.numpy_mask(n)
        for BQ, A in TILES:
            bi = blockindex.build(m, N, BQ, A)
            pred, _ = pcost.cost(m, N, BQ, A, exact_only=True)
            if bi.elements != pred:
                fails.append(f"{n} {BQ}x{A}: index {bi.elements} != model {pred}")
        print(f"  {n:<16} OK across {len(TILES)} tile shapes")

    print("\n=== 3. Triton kernel vs dense reference ===")
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    for n in NAMES:
        m = masks_gpu.numpy_mask(n)
        ref = reference.attention_reference(q, k, v, masks_gpu.dense_bool(n, N))
        worst = 0.0
        for BQ, A in TILES:
            bi = blockindex.build(m, N, BQ, A)
            kvi, kvn, kvp = blockindex.to_cuda(bi)
            kind, p0, p1, p2 = masks_gpu.triton_params(n)
            out = block_sparse_attention(q, k, v, kvi, kvn, kvp, kind, p0, p1, p2, BQ, A)
            worst = max(worst, reference.max_abs_err(out, ref))
        print(f"  {n:<16} max abs err {worst:.2e}  {'OK' if worst < TOL else 'FAIL'}")
        if worst >= TOL:
            fails.append(f"kernel mismatch: {n} err {worst:.2e}")

    print("\n=== 4. class A permuted path vs the same reference ===")
    for n, s in (("dilated-4", 4), ("dilated-8", 8)):
        m = masks_gpu.numpy_mask(n)
        ref = reference.attention_reference(q, k, v, masks_gpu.dense_bool(n, N))
        p_np = permute.residue_perm(N, s)
        perm = torch.from_numpy(p_np).cuda()
        qp, kp, vp = (permute.apply_perm(x, perm) for x in (q, k, v))

        M = np.stack([m.row_cols(i, N) for i in range(N)])[p_np][:, p_np]
        bi = blockindex.build(_Dense(M), N, 16, 16)
        kvi, kvn, kvp = blockindex.to_cuda(bi)
        kind, p0, p1, p2 = masks_gpu.triton_params(n)
        outp = block_sparse_attention(qp, kp, vp, kvi, kvn, kvp, kind, p0, p1, p2,
                                      16, 16, perm_q=perm.int(), perm_kv=perm.int())
        out = outp.index_select(1, permute.invert(perm).long())
        err = reference.max_abs_err(out, ref)
        print(f"  {n:<16} waste {bi.waste:.3f}  max abs err {err:.2e}"
              f"  {'OK' if err < TOL else 'FAIL'}")
        if err >= TOL:
            fails.append(f"permuted mismatch: {n} err {err:.2e}")

    print("\n" + ("ALL OK" if not fails else f"{len(fails)} FAILURES"))
    for f in fails:
        print("  -", f)
    return 1 if fails else 0


class _Dense:
    """Wrap an explicit boolean matrix in the polyattn mask interface."""
    def __init__(self, M):
        self.M, self.name = M, "permuted"
    def row_cols(self, q, N):
        return self.M[q]


if __name__ == "__main__":
    raise SystemExit(main())
