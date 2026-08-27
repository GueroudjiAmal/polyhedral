"""EXPERIMENT 0 -- THE GO/NO-GO. Run this and nothing else, after the smoke test.

One question: does a 7.79x predicted element reduction become a wall-clock win?

dilated-8 + residue-perm-8 is the case where the analysis predicts the largest
effect, and it is class A, so the permutation is a one-time per-layer cost rather
than a per-tile gather. If this does not move, mechanism 1 is dead and no amount
of further analysis rescues it. If it does move, everything else in gpu/ becomes
worth running.

Deliberately narrow: one mask, one sequence length, one head count, forward only.
A broad sweep here would bury the one number that matters.

Baselines, both reported, because whichever one is omitted is the one a reviewer
will ask for:
  * FlexAttention at its DEFAULT BlockMask (128x128) -- what people actually run
  * FlexAttention at its BEST over a small block-size sweep -- the honest opponent
  * our kernel unpermuted, so the comparison isolates the transform from the kernel

The permutation is timed SEPARATELY and never folded in.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute, reference     # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

NAME, STRIDE = "dilated-8", 8
N, D, B, H = 4096, 64, 1, 8
FLEX_BLOCKS = (128, 64, 32)
OUR_TILES = ((128, 128), (16, 16))
# exp0 asks whether small tiles lose on occupancy -- so launch config, the knob
# that decides that, must not be pinned. A fixed num_warps favours the large tile
# and would measure our configuration rather than the hardware.
LAUNCH = ((4, 2), (8, 2), (4, 3), (8, 3), (2, 2))


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def best_launch(fn_of):
    """Min over a small launch-config sweep. Returns (ms, num_warps, num_stages)."""
    best = (float("inf"), None, None)
    for w, st in LAUNCH:
        try:
            t = bench.time_ms(lambda: fn_of(w, st), warmup=10, reps=40)[0]
        except Exception:
            continue                       # config exceeds resources for this tile
        if t < best[0]:
            best = (t, w, st)
    return best


def main():
    torch.manual_seed(0)
    q4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    k4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    v4 = torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
    q3, k3, v3 = (x.reshape(B * H, N, D).contiguous() for x in (q4, k4, v4))

    m = masks_gpu.numpy_mask(NAME)
    kind, p0, p1, p2 = masks_gpu.triton_params(NAME)
    M = np.stack([m.row_cols(i, N) for i in range(N)])
    p_np = permute.residue_perm(N, STRIDE)
    perm = torch.from_numpy(p_np).cuda()
    Mp = M[p_np][:, p_np]
    qp, kp, vp = (permute.apply_perm(x, perm) for x in (q3, k3, v3))

    # correctness gate, again, here -- a wrong permuted kernel would look fast
    ref = reference.attention_reference(q3, k3, v3, masks_gpu.dense_bool(NAME, N))
    bi_p = blockindex.build(_Dense(Mp), N, 16, 16)
    idx_p = blockindex.to_cuda(bi_p)
    outp = block_sparse_attention(qp, kp, vp, *idx_p, kind, p0, p1, p2, 16, 16,
                                  perm_q=perm.int(), perm_kv=perm.int())
    err = reference.max_abs_err(outp.index_select(1, permute.invert(perm).long()), ref)
    print(f"permuted-path correctness: max abs err {err:.2e}"
          f"  {'OK' if err < 3e-3 else 'FAIL -- STOP HERE'}")
    if err >= 3e-3:
        return 1

    rows = []
    base_el = base_ms = None
    for BQ, A in OUR_TILES:
        bi = blockindex.build(_Dense(M), N, BQ, A)
        idx = blockindex.to_cuda(bi)
        t, w, st = best_launch(lambda w, st: block_sparse_attention(
            q3, k3, v3, *idx, kind, p0, p1, p2, BQ, A, num_warps=w, num_stages=st))
        if base_ms is None:
            base_ms, base_el = t, bi.elements
        rows.append([f"ours {BQ}x{A} identity", bi.waste, t, f"w{w}/s{st}",
                     base_el / bi.elements, base_ms / t])
    t_perm, wp, stp = best_launch(lambda w, st: block_sparse_attention(
        qp, kp, vp, *idx_p, kind, p0, p1, p2, 16, 16,
        perm_q=perm.int(), perm_kv=perm.int(), num_warps=w, num_stages=st))
    rows.append(["ours 16x16 residue-perm-8", bi_p.waste, t_perm, f"w{wp}/s{stp}",
                 base_el / bi_p.elements, base_ms / t_perm])

    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        fa = torch.compile(flex_attention, dynamic=False)
        mod = masks_gpu.flex_mask_mod(NAME)
        for bs in FLEX_BLOCKS:
            try:
                bm = create_block_mask(mod, B=None, H=None, Q_LEN=N, KV_LEN=N,
                                       BLOCK_SIZE=bs, _compile=True)
                t = bench.time_ms(lambda: fa(q4, k4, v4, block_mask=bm), reps=50)[0]
                rows.append([f"FlexAttention {bs}x{bs}"
                             + ("  (default)" if bs == 128 else ""), float("nan"),
                             t, "-", float("nan"), base_ms / t])
            except Exception as e:
                rows.append([f"FlexAttention {bs}x{bs}", float("nan"),
                             f"FAIL {type(e).__name__}", "-", float("nan"),
                             float("nan")])
    except Exception as e:
        print(f"flex_attention unavailable: {e}")

    pc = permute.permutation_cost_ms(q3, k3, v3, perm)
    bench.report(rows, [("implementation", 28), ("waste", 9), ("ms", 11),
                        ("launch", 9), ("PREDICTED", 11), ("MEASURED", 11)],
                 title=f"GO/NO-GO   {NAME} + residue-perm-{STRIDE}   "
                       f"N={N} B={B} H={H} D={D}",
                 note=f"one-time permutation of Q/K/V: {pc:.3f} ms, NOT included above\n"
                      "PREDICTED and MEASURED are both speedups over the 128x128 "
                      "unpermuted baseline, printed adjacent on purpose:\n"
                      "  they agree      -> the cost model predicts hardware\n"
                      "  measured >> 1 but predicted ~ 1 -> the win is not the transform\n"
                      "  predicted >> measured -> the model overprices what it counts\n"
                      "ms is the best over a launch-config sweep, so a slow small tile "
                      "is the hardware and not our configuration.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
