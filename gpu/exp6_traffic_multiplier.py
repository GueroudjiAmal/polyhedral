"""EXPERIMENT 6 -- the only measurement that can move a selection decision.

Both proposed hardware corrections turned out to be argmin-inert:

  * a multiplicative tile penalty p(BQ,A) multiplies every candidate equally at
    fixed tile shape;
  * a per-CTA fixed cost F enters as an ADDITIVE constant, because the grid is
    (N/BQ, BH) and nprog does not depend on the transform.

Verified numerically at 0/480 and 0/324 cells respectively. So to move a
decision, a term must differ BY TRANSFORM at a fixed tile shape -- and going
through the taxonomy, every transform-dependent quantity is MEMORY, not compute:

    identity / residue-perm-8   16 kv rows per 16x16 tile   (class A, contiguous)
    shear                       31                          (class B, per-tile gather)
    stridefold-8               136                          (class B, worst case)

Element counting measures compute. Therefore element counting cannot distinguish
class A from class B EVEN IN PRINCIPLE, and the class A/B taxonomy -- ranked
THIRD in novelty behind selection and the impossibility argument -- is the only
axis on which a hardware-informed selector can beat a counting one.

THE MEASUREMENT. Identical mask, identical tiles, identical element count,
identical output -- the only difference is whether K/V rows are read contiguously
from a physically permuted tensor (class A) or gathered per tile from the
original (what class B must do). The ratio IS the traffic multiplier, with
nothing else varying.

Reported against the flip thresholds two other sessions computed independently
on their own instance sets: 1.20x flips 249/544 contested cases, 1.50x flips
415/544, 1.94x flips 510/544.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute, reference     # noqa: E402

from polyattn import transforms                             # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

N, D, BH = 4096, 64, 8
CASES = [("dilated-8", 8), ("dilated-4", 4), ("local256+str8", 8)]
_ALL_CASES = list(CASES)
TILES = [(128, 128), (128, 32), (64, 64), (32, 32), (16, 16)]
#: exp6's gather touches exactly A rows, so on its own it isolates COALESCING at a
#: constant row count. Class B touches A + a*(BQ-1) DISTINCT rows -- 1.94x to
#: 15.5x more, depending on tile shape (polyattn.transforms.kv_per_tile). Sweeping
#: R/A turns the lower bound into a curve: penalty(coalescing) x penalty(R/A)
#: predicts any class B configuration without implementing shear.
ROW_MULTS = (2, 4, 8, 16)   # 16x16's upper bound is 16x, 128x128's is 128x --
                            # a sweep stopping at 8 does not reach the top


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3))
    rows = []
    for name, s in CASES:
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        M = np.stack([m.row_cols(i, N) for i in range(N)])
        p_np = permute.residue_perm(N, s)
        perm = torch.from_numpy(p_np).cuda()
        pi = perm.int()
        Mp = M[p_np][:, p_np]
        qp = permute.apply_perm(q, perm)
        kp, vp = permute.apply_perm(k, perm), permute.apply_perm(v, perm)

        for BQ, A in TILES:
            bi = blockindex.build(_Dense(Mp), N, BQ, A)
            idx = blockindex.to_cuda(bi)
            # contiguous: K/V physically permuted, rows read in order
            out_c = block_sparse_attention(qp, kp, vp, *idx, kind, p0, p1, p2,
                                           BQ, A, perm_q=pi, perm_kv=pi)
            # gathered: K/V left in place, rows gathered per tile
            out_g = block_sparse_attention(qp, k, v, *idx, kind, p0, p1, p2,
                                           BQ, A, perm_q=pi, perm_kv=pi,
                                           gather_kv=True)
            err = reference.max_abs_err(out_c, out_g)
            if err > 1e-3:
                rows.append([f"{name} {BQ}x{A}", bi.elements, float("nan"),
                             float("nan"), f"OUTPUTS DIFFER {err:.1e}"])
                continue
            t_c, sd_c = bench.time_ms(lambda: block_sparse_attention(
                qp, kp, vp, *idx, kind, p0, p1, p2, BQ, A,
                perm_q=pi, perm_kv=pi))
            t_g, sd_g = bench.time_ms(lambda: block_sparse_attention(
                qp, k, v, *idx, kind, p0, p1, p2, BQ, A,
                perm_q=pi, perm_kv=pi, gather_kv=True))
            mult = t_g / t_c
            curves = {}
            for scat in (False, True):
                cur = []
                for R in ROW_MULTS:
                    try:
                        t_r = bench.time_ms(lambda: block_sparse_attention(
                            qp, k, v, *idx, kind, p0, p1, p2, BQ, A, perm_q=pi,
                            perm_kv=pi, gather_kv=True, gather_mult=R,
                            gather_scatter=scat), reps=30)[0]
                        cur.append(f"{R}:{t_r/t_c:.2f}")
                    except Exception as e:
                        cur.append(f"{R}:{type(e).__name__[:6]}")
                curves["scat" if scat else "cont"] = " ".join(cur)
            lo = transforms.kv_per_tile("B", 1, 1, BQ, A) / A
            hi = BQ * A / A
            rows.append([f"{name} {BQ}x{A}",
                         f"{bi.elements}  cv {max(sd_c/t_c, sd_g/t_g)*100:.1f}%",
                         f"{mult:.2f}x",
                         f"[{lo:.1f}x..{hi:.0f}x]",
                         "cont " + curves["cont"] + " | scat " + curves["scat"]])

    bench.report(rows, [("case", 22), ("elements", 12), ("coalesce", 10),
                        ("class B bracket", 18), ("penalty(R/A): contiguous | scattered", 60)],
                 title=f"class A contiguous vs class B gather, MATCHED elements   "
                       f"N={N} BH={BH} D={D}",
                 note="Both columns compute the SAME tiles and produce the SAME output --\n"
                      "verified per row before timing. The ratio is the traffic term the\n"
                      "element-count model cannot see, and it is the only quantity found\n"
                      "so far that differs by transform at a fixed tile shape.\n"
                      "FIRST NUMBER = coalescing only, at A rows. It is a LOWER BOUND on\n"
                      "the class B penalty, NOT the penalty: class B touches 1.94x-15.5x\n"
                      "more rows depending on tile shape. A small first number does NOT\n"
                      "close the question -- that inference was proposed and is wrong.\n"
                      "The R multipliers sweep the row-count axis so the two factors can\n"
                      "be composed for any class B configuration without implementing it.")
    return 0


if __name__ == "__main__":
    # `python exp6_traffic_multiplier.py dilated-8` runs one mask. exp6 is by far
    # the heaviest experiment -- GATHER_MULT x GATHER_SCATTER multiply the Triton
    # constexpr space 9x per tile shape, so all three masks is ~135 kernel
    # compiles and does not fit a debug-queue hour. One mask is ~45 and does.
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        CASES = [c for c in CASES if c[0] in want]
        if not CASES:
            print(f"no such mask; choose from "
                  f"{[c[0] for c in globals()['_ALL_CASES']]}")
            raise SystemExit(1)
        print(f"running only {sorted(want)}\n")
    raise SystemExit(main())
