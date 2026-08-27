"""EXPERIMENT 9 -- an out-of-sample test of the fixed-cost account.

The class A transform predicted 7.79x fewer elements and measured 3.34x. That
57% miss has been explained twice by a per-CTA fixed-cost account, and an
explanation invoked twice without being tested is an excuse. This tests it.

THE MODEL.  time = nprog*F + tiles*V, with nprog = (N/BQ) transform-independent.
Writing rho for the element ratio and phi = (F/V) / tiles_per_program_of_the_fast
variant:

    measured_ratio = (phi + rho) / (phi + 1)

phi -> 0 gives no compression; phi -> inf gives 1. So the knob is TILES PER
PROGRAM OF THE FAST VARIANT, and the test is a cell where that is large while rho
stays large.

CALIBRATION, from the miss itself -- one parameter, one point:
    dilated-8, N=4096, 16x16:  rho 7.79, measured 3.34
    -> phi = (rho-m)/(m-1) = 1.90 ;  fast variant has 16.5 tiles/program
    -> F/V = 31.4 tile-equivalents of fixed cost per program

THE OUT-OF-SAMPLE CELL, verified against this repo's own counts:
    dilated-2, N=16384, 16x16, {identity, residue-perm-2}
      identity 524,800 tiles | rp2 262,656 | rho = 2.00
      fast variant: 256.5 tiles/program  ->  phi = 0.12
      PREDICTS measured 1.89, i.e. 95% of the element ratio

So the same one-parameter model predicts a 5% shortfall here against the 57% it
must explain at dilated-8 -- a 10x difference in predicted shortfall, which makes
this a real out-of-sample prediction rather than a refit.

  PASS  1.80-1.98: fixed cost survives as a MAGNITUDE account. (It is already
        dead as a SELECTOR -- an additive constant at fixed tile shape leaves
        every argmin alone, 0/480 cells.)
  FAIL  near 1.5 or below: a large shortfall persists where the model predicts
        almost none. Both halves are then gone and the 57% miss has NO
        explanation, which puts the whole element-count-to-wall-clock mapping in
        question rather than just its magnitude.

THIS IS A ONE-SIDED TEST AND MUST BE WRITTEN UP AS ONE. Because dilated-2's fast
variant has 256.5 tiles per program, phi is small for ANY plausible F/V, so the
model predicts "almost no shortfall" across a very wide parameter range:

    in-job d8  |  2.50   3.00   3.34   4.00   5.00   6.00
    F/V        |  58.2   39.5   31.4   20.8   11.5    5.9
    predicts   |  1.813  1.865  1.889  1.923  1.955  1.976

A 2.4x swing in the calibration -- an order of magnitude wider than the 23%
cross-job variance -- moves the prediction only from 1.81 to 1.98. So this cell
CAN KILL the account and CANNOT MEASURE it: landing near 1.89 is consistent with
F/V anywhere from 6 to 58.

To PIN the constant you need intermediate phi, near 1. The `pin` row below is
dilated-4 at N=8192, 16x16, where phi spans 0.18..0.90 over that same F/V range
and the prediction moves 3.53 / 3.00 / 2.56 -- there the answer discriminates.

Timed with bench.paired_time: a predicted 5% shortfall cannot be measured by
timing one variant fully and then the other.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute                # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

D, BH = 64, 8
FV = 31.4                       # fitted at dilated-8; NOT refitted below
#: (label, mask, stride, N, BQ, A). The calibration row runs FIRST so F/V can be
#: refitted in-job and both predictions printed.
CELLS = [
    ("calibration (in-sample; refits F/V)", "dilated-8", 8, 4096, 16, 16),
    ("out-of-sample FALSIFIER (phi small)", "dilated-2", 2, 16384, 16, 16),
    ("PIN the constant (phi ~ 0.2-0.9)",    "dilated-4", 4,  8192, 16, 16),
    ("fallback, smaller",                   "dilated-2", 2,  8192, 32, 32),
]


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def main():
    torch.manual_seed(0)
    fv_fit = None            # refitted from the in-job calibration row
    print(f"{'cell':<38}{'rho':>7}{'tpp':>7}{'pred(31.4)':>11}"
          f"{'pred(refit)':>12}{'measured':>11}{'verdict':>10}")
    print("-" * 98)
    for label, name, s, N, BQ, A in CELLS:
        try:
            m = masks_gpu.numpy_mask(name)
            kind, p0, p1, p2 = masks_gpu.triton_params(name)
            M0 = np.stack([m.row_cols(i, N) for i in range(N)])
            p_np = permute.residue_perm(N, s)
            perm = torch.from_numpy(p_np).cuda()
            Mp = M0[p_np][:, p_np]

            bi_i = blockindex.build(_Dense(M0), N, BQ, A)
            bi_p = blockindex.build(_Dense(Mp), N, BQ, A)
            rho = bi_i.elements / bi_p.elements
            tpp = (bi_p.elements / (BQ * A)) / (N // BQ)
            def _pred(fv):
                ph = fv / tpp
                return (ph + rho) / (ph + 1)
            pred = _pred(FV)

            q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
                       for _ in range(3))
            qp, kp, vp = (permute.apply_perm(x, perm) for x in (q, k, v))
            ii, ip = blockindex.to_cuda(bi_i), blockindex.to_cuda(bi_p)
            pi = perm.int()

            fa_ = lambda: block_sparse_attention(q, k, v, *ii, kind, p0, p1, p2,
                                                 BQ, A, num_warps=1, num_stages=3)
            fb_ = lambda: block_sparse_attention(qp, kp, vp, *ip, kind, p0, p1, p2,
                                                 BQ, A, perm_q=pi, perm_kv=pi,
                                                 num_warps=1, num_stages=3)
            ta, tb, ratio_bp, rsd = bench.paired_time(fb_, fa_, warmup=15, reps=80)
            meas = ratio_bp            # slow / fast, paired

            if label.startswith("calibration"):
                # REFIT IN-JOB. The out-of-sample property comes from fitting on
                # THIS cell and testing on a DIFFERENT one -- different mask, N,
                # rho and tiles/program -- not from which job supplied the
                # number. So a differing calibration is a reason to refit, NOT to
                # void the comparison: refitting makes both cells same-job and
                # immune to the 23% swing rather than merely guarded against it.
                # Refitting from the TEST cell would be illegitimate. Nobody is
                # proposing that, and this comment exists so no later reader
                # conflates the two.
                fv_fit = ((rho - meas) / (meas - 1)) * tpp if meas > 1 else None
                pr_ = f"{fv_fit:.1f}" if fv_fit else "n/a"
                print(f"{label + '  ' + name:<38}{rho:>7.2f}{tpp:>7.1f}"
                      f"{pred:>11.2f}{'refit F/V=' + pr_:>12}"
                      f"{meas:>8.2f}+-{rsd:.2f}{'':>10}")
                continue

            pred_r = _pred(fv_fit) if fv_fit else float("nan")
            band = (min(pred, pred_r) if fv_fit else pred, 
                    max(pred, pred_r) if fv_fit else pred)
            v_ = ("PASS" if 0.95 * band[0] <= meas <= 1.03 * band[1] else
                  "FAIL" if meas < 0.8 * band[0] else "marginal")
            print(f"{label + '  ' + name:<38}{rho:>7.2f}{tpp:>7.1f}{pred:>11.2f}"
                  f"{pred_r:>12.2f}{meas:>8.2f}+-{rsd:.2f}{v_:>10}")
        except Exception as e:
            print(f"{label + '  ' + name:<46}  FAILED {type(e).__name__}: "
                  f"{str(e).splitlines()[0][:60]}")
    print("""
rho = element ratio (what counting predicts) | tpp = tiles per program, fast variant
phi = (F/V)/tpp with F/V = 31.4 fitted ONCE at dilated-8 and never refitted
pred = (phi+rho)/(phi+1) -- the fixed-cost account's out-of-sample prediction
measured = PAIRED interleaved ratio, so a 5% effect is not lost in drift

pred(31.4) uses the constant fitted in an EARLIER job; pred(refit) uses this
job's own calibration row. Both are printed so a reader can see the test does not
depend on the choice -- a 2.4x swing in the calibration moves the falsifier's
prediction only 1.81 -> 1.98.

Read the FALSIFIER row one-sidedly: at or below ~1.5 the fixed-cost account is
dead. Landing near 1.89 does NOT confirm it, only fails to kill it. The PIN row
is where the constant is actually measured -- its prediction moves 3.53/3.00/2.56
across the same F/V range, so there the answer discriminates.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
