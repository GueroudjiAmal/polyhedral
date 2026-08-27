"""EXPERIMENT 7 -- the enumeration was incomplete, and the gap is COMPUTE-side.

Two sessions concluded that every transform-dependent hardware term is memory,
and therefore that the class A/B taxonomy is the only axis element counting
cannot see. The second clause is an enumeration and it is false.

LOAD IMBALANCE. The grid is one program per row-block, so runtime tracks the
SLOWEST program, not the sum. `max tiles per row-block` is transform-dependent
and is NOT a monotone function of total tiles -- so it escapes the
argmin-inertness proof, which covers only g(elements, BQ, A) increasing in
elements.

Verified here on the analytical side: 17 of 160 cells have a different argmin
under total tiles than under max-tiles-per-program, all on UNION masks, none on
pure lattices. Both candidates are class A, so a memory account predicts no
difference whatsoever.

THE ADJUDICATION. One case where the two criteria disagree in DIRECTION, so a
single pair of timings settles it:

    local256+str8, N=2048, 128x32
      counting  -> rp2   (408 tiles total, 40 max per program)   rp2 by 1.04x
      makespan  -> rp4   (424 tiles total, 34 max per program)   rp4 by 1.18x

PREDICTION ON THE RECORD, BEFORE THE RUN: rp4 wins on wall-clock despite
computing MORE total elements. If rp2 wins, makespan is not the operative term at
this configuration and the counting model survives this attack.

CAVEAT, also on the record: pure makespan is the wrong model when programs
greatly outnumber SMs. At N=2048, BQ=128, BH=8 that is 128 programs on ~108 SMs,
roughly one wave, where imbalance bites hardest. At BQ=16 it is 1024 programs and
~9 waves and it should largely average out. So the 16x16 rows below are the
control: the effect should WEAKEN there. If it does not, the explanation is
something other than imbalance.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute                # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

NAME = "local256+str8"
D, BH = 64, 8
CASES = [(2048, 128, 32), (2048, 128, 16), (1024, 16, 16), (2048, 16, 16)]
#: The candidate set. A mask has a disagreement to adjudicate only if the set
#: CONTAINS the transforms that disagree -- sinks at 128x128 has nothing to
#: measure under {identity, rp2, rp4}, because the disagreement needs rp8. Quote
#: this set with any number this experiment produces.
FOLDS = (2, 4, 8)


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def main():
    torch.manual_seed(0)
    kind, p0, p1, p2 = masks_gpu.triton_params(NAME)
    m = masks_gpu.numpy_mask(NAME)
    rows = []
    for N, BQ, A in CASES:
        q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3))
        M0 = np.stack([m.row_cols(i, N) for i in range(N)])
        nprog, waves = (N // BQ) * BH, (N // BQ) * BH / 108.0
        best_tot = best_max = None
        rec = {}
        for s in FOLDS:
            p_np = permute.residue_perm(N, s)
            perm = torch.from_numpy(p_np).cuda()
            Mp = M0[p_np][:, p_np]
            t = Mp.reshape(N // BQ, BQ, N // A, A).any(axis=(1, 3))
            tot, mx = int(t.sum()), int(t.sum(axis=1).max())
            bi = blockindex.build(_Dense(Mp), N, BQ, A)
            idx = blockindex.to_cuda(bi)
            qp, kp, vp = (permute.apply_perm(x, perm) for x in (q, k, v))
            ms, sd = bench.time_ms(lambda: block_sparse_attention(
                qp, kp, vp, *idx, kind, p0, p1, p2, BQ, A,
                perm_q=perm.int(), perm_kv=perm.int()))
            rec[f"rp{s}"] = (tot, mx, ms, sd)
            if best_tot is None or tot < rec[best_tot][0]:
                best_tot = f"rp{s}"
            if best_max is None or mx < rec[best_max][1]:
                best_max = f"rp{s}"
        best_ms = min(rec, key=lambda kk: rec[kk][2])
        verdict = ("COUNTING" if best_ms == best_tot and best_tot != best_max else
                   "MAKESPAN" if best_ms == best_max and best_tot != best_max else
                   "no disagreement to adjudicate" if best_tot == best_max else
                   f"neither ({best_ms})")
        rows.append([f"N={N} {BQ}x{A}", f"{waves:.1f}", best_tot, best_max,
                     best_ms, verdict])
        for tag, (tot, mx, ms, sd) in rec.items():
            rows.append([f"    {tag}", "", f"tot {tot}", f"max {mx}",
                         f"{ms:.3f} ms", f"cv {sd/ms*100:.1f}%"])

    bench.report(rows, [("case", 16), ("waves", 7), ("argmin tot", 12),
                        ("argmin max", 12), ("argmin time", 12), ("verdict", 32)],
                 title=f"{NAME}: does load imbalance move the argmin?   BH={BH} D={D}",
                 note=f"CANDIDATE SET: {{{', '.join(f'rp{s}' for s in FOLDS)}}} -- a disagreement\n"
                      "exists only if the set contains the transforms that disagree.\n"
                      "Both candidates are class A, so a memory-only account predicts NO\n"
                      "difference. Prediction on record: MAKESPAN wins at 128x32 (~1 wave),\n"
                      "and the effect WEAKENS at 16x16 (~9 waves), which is the control.\n"
                      "If 16x16 shows the same effect, the cause is not imbalance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
