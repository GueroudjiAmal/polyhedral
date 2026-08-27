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
#: Widened after the first run: BOTH 16x16 rows picked w2/s2, the minimum of
#: both parameters in the original sweep ((4,2),(8,2),(4,3),(8,3),(2,2)). A winner
#: on the boundary means the optimum may lie outside what was offered, and both
#: headline numbers -- the 1.61x small-tile penalty and the 3.19x transform gain --
#: came from those rows. num_warps=1 is legal and was never tried.
LAUNCH = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4),
          (4, 1), (4, 2), (4, 3), (8, 2), (8, 3))

#: num_warps cannot go below 1, so a winner at w1 is a HARDWARE FLOOR, not a
#: sweep that was too narrow. num_stages CAN go to 1, so s2 as the minimum was a
#: genuine gap -- run 2 reported "ON SWEEP BOUNDARY (warps)" for both 16x16 rows,
#: which was a false alarm, while the real gap (stages) went unflagged on the
#: 128x128 row. The detector below distinguishes them.
_HARD_FLOOR = {"warps": 1, "stages": 1}


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def _warn_if_on_boundary(cfg, label=""):
    """A winner at the edge of the sweep is a sweep that was too narrow.

    This is what the first run did silently: both 16x16 rows chose the minimum
    num_warps AND the minimum num_stages offered, and nothing said so.
    """
    if cfg is None:
        return ""
    w, st = (int(x[1:]) for x in cfg.split("/"))
    ws = sorted({c[0] for c in LAUNCH})
    ss = sorted({c[1] for c in LAUNCH})
    edge = []
    for n, v, lo, hi in (("warps", w, ws[0], ws[-1]), ("stages", st, ss[0], ss[-1])):
        if v == hi:
            edge.append(f"{n} at sweep max")
        elif v == lo and lo > _HARD_FLOOR[n]:
            edge.append(f"{n} at sweep min, floor is {_HARD_FLOOR[n]}")
        # v == lo == hard floor: the hardware limit, not a narrow sweep
    return f"  <-- SWEEP TOO NARROW ({'; '.join(edge)}){label}" if edge else ""


#: The sweep used in run 1, kept so the widening can be measured WITHIN a job.
#: Comparing run 1's numbers against run 2's cannot separate tuning from noise:
#: rp8 went 0.122 -> 0.139 between jobs even though w2/s2 was in both sweeps, so
#: that 14% is node/job variance and the apparent effect of widening is inside it.
NARROW = ((4, 2), (8, 2), (4, 3), (8, 3), (2, 2))


def best_launch(fn_of, configs=LAUNCH):
    """Min over a launch-config sweep. Returns (ms, stdev, num_warps, num_stages)."""
    best = (float("inf"), 0.0, None, None)
    for w, st in configs:
        try:
            t, sd = bench.time_ms(lambda: fn_of(w, st), warmup=10, reps=60)
        except Exception:
            continue                       # config exceeds resources for this tile
        if t < best[0]:
            best = (t, sd, w, st)
    return best


def _why(e, n=3):
    """Root cause, not just the wrapper class. BackendCompilerFailed on its own
    says nothing -- two runs reported it and neither told us why."""
    cur, out = e, []
    for _ in range(6):
        first = (str(cur).strip().splitlines() or [""])[0]
        out.append(f"{type(cur).__name__}: {first[:90]}" if first else type(cur).__name__)
        nxt = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return " <- ".join(out[:n])


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
        fn = lambda w, st: block_sparse_attention(
            q3, k3, v3, *idx, kind, p0, p1, p2, BQ, A, num_warps=w, num_stages=st)
        t, sd, w, st = best_launch(fn)
        t_nar = best_launch(fn, NARROW)[0]
        if base_ms is None:
            base_ms, base_el = t, bi.elements
        cfg = f"w{w}/s{st}"
        rows.append([f"ours {BQ}x{A} identity" + _warn_if_on_boundary(cfg),
                     bi.waste, f"{t:.4f} cv{sd/t*100:.0f}% (narrow {t_nar:.4f})",
                     cfg, base_el / bi.elements, base_ms / t])
    fnp = lambda w, st: block_sparse_attention(
        qp, kp, vp, *idx_p, kind, p0, p1, p2, 16, 16,
        perm_q=perm.int(), perm_kv=perm.int(), num_warps=w, num_stages=st)
    t_perm, sdp, wp, stp = best_launch(fnp)
    t_perm_nar = best_launch(fnp, NARROW)[0]
    cfgp = f"w{wp}/s{stp}"
    rows.append(["ours 16x16 residue-perm-8" + _warn_if_on_boundary(cfgp),
                 bi_p.waste,
                 f"{t_perm:.4f} cv{sdp/t_perm*100:.0f}% (narrow {t_perm_nar:.4f})",
                 cfgp, base_el / bi_p.elements, base_ms / t_perm])

    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
        mod = masks_gpu.flex_mask_mod(NAME)
        # FlexAttention gets a FRESH compile per configuration, with a dynamo reset
        # between them. Reusing one compiled callable across configs measured the
        # same dilated-8 baseline at 0.322 ms in exp0 and 0.3953 ms in the
        # diagnostic -- 23% apart -- so the reused path was not measuring what it
        # claimed. Only the WITHIN-JOB ratio against our kernel is quotable.
        def _flex_time(mod, bs, kopt):
            torch._dynamo.reset()
            fa = torch.compile(flex_attention, dynamic=False)
            bm = create_block_mask(mod, B=None, H=None, Q_LEN=N, KV_LEN=N,
                                   BLOCK_SIZE=bs, _compile=True)
            return bench.time_ms(lambda: fa(q4, k4, v4, block_mask=bm, **kopt),
                                 warmup=25, reps=100)
        for bs in FLEX_BLOCKS:
            try:
                # Sub-128 is STRUCTURALLY IMPOSSIBLE in torch 2.6.0:
                # kernel_options BLOCK_M/BLOCK_N are dropped before the lowering
                # (the options dict reaching inductor contains only the four
                # boolean flags), so the divisibility check always compares
                # against the default 128. Retained as a regression probe -- if a
                # future torch plumbs them through, this row starts passing.
                kopt = {} if bs >= 128 else {"kernel_options": {"BLOCK_M": bs,
                                                                "BLOCK_N": bs}}
                t, sd = _flex_time(mod, bs, kopt)
                rows.append([f"FlexAttention {bs}x{bs}"
                             + ("  (only size this torch supports)" if bs == 128
                                else ""), float("nan"),
                             f"{t:.4f} cv{sd/t*100:.0f}%", "-", float("nan"),
                             base_ms / t])
            except Exception as e:
                rows.append([f"FlexAttention {bs}x{bs}", float("nan"),
                             f"FAIL {_why(e)}", "-", float("nan"),
                             float("nan")])
    except Exception as e:
        print(f"flex_attention unavailable: {e}")

    pc = permute.permutation_cost_ms(q3, k3, v3, perm)
    bench.report(rows, [("implementation", 52), ("waste", 9),
                        ("ms  cv  (narrow sweep)", 30),
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
