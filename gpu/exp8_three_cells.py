"""EXPERIMENT 8 -- the three cells that measure a COEFFICIENT rather than a rate.

Today produced six tables of disagreement RATES. None of them constrains the
claim, because a rate answers "how often could hardware disagree with counting"
and the claim needs "by how much, and does it". These three cells measure the
coefficient. They are the only ones worth queue time.

  1  sinks4+win256   N=1024  128x128  {identity, rp8}
       identity 26 tiles / 12 runs | rp8 64 tiles / 8 runs
       counting -> identity ; contiguity -> rp8
       flips iff c_runs/c_T > 9.50   -- extreme end of the bracket
  2  local256+str8   N=1024  128x128  {identity, rp2}
       identity 36 tiles / 8 runs | rp2 34 tiles / 15 runs
       counting -> rp2 ; contiguity -> identity
       flips iff c_runs/c_T > 0.29
  3  local256+str8   N=2048  128x32   {identity, rp2, rp4}
       rp2 408 tiles / max 40 | rp4 424 / max 34
       DIRECTION ONLY, AND THE BOUNDARY IS 1.0.
         wave account  rp4's longest job is shorter -> rp4 faster -> ratio < 1
         counting      rp4 visits more tiles        -> rp4 slower -> ratio > 1
       NEITHER MAGNITUDE IS TRUSTWORTHY, so no midpoint between them means
       anything. "rp4 by 1.18x" was a ratio of two LOWER bounds, not a speedup.
       And 424/408 = 1.039 assumes time is proportional to tile count -- the
       model exp0 falsified, predicting 8.00x and measuring 3.18x. What exp0 DOES
       support is that counting got the DIRECTION right while overshooting the
       size 2.5x, which is exactly why direction is the falsifier and magnitude
       is not. rp4 has both a shorter longest job and a work term below it, so
       under ANY model where makespan is governed by the longest job, rp4 wins.
       If rp2 wins, that account is wrong outright with nothing to retreat to.

1 and 2 BRACKET the contiguity coefficient from both sides. 3 tests the wave
account with NO FREE PARAMETER -- if rp2 wins there, the makespan story is wrong
outright and there is no coefficient to retreat behind.

N IS LOAD-BEARING IN CELL 3 AND REPRODUCING IT AT A CONVENIENT N GIVES A NULL.
Verified analytically: at N=2048 it is 128 programs on 108 SMs (1.19 waves) and
counting picks rp2 while makespan picks rp4. At N=1024 it is 0.59 waves AND rp4
loses on BOTH criteria -- there is no disagreement to measure at all. The tile
counts are asserted at runtime below so a silent drift shows up as a failure
rather than as a null result.

RESOLUTION. Every cell reports median and spread, and the spread decides whether
the direction is resolved -- there is no pre-computed noise threshold, because any
such threshold would have to be derived from the same point predictions this
experiment declines to trust. Cells 2 and 3 have margins near the noise floor. These are not exp0-sized effects.
Each row reports the median AND the spread, and says outright when the difference
is inside the noise -- "could not resolve" is an honest answer and a false
negative is not. Cell 1 is 26 vs 64 tiles against 12 vs 8 runs, so it should be
unambiguous whichever way it lands, which is why it runs first.
"""
import sys

from collections import namedtuple as _NT

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute                # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

D, BH = 64, 8
from launchsweep import LAUNCH, _HARD_FLOOR, _warn_if_on_boundary  # noqa: E402

CELLS = [
    ("1  sinks4+win256", "sinks4+win256", 1024, 128, 128, (None, 8),
     {"identity": (26, 12), "rp8": (64, 8)}, "counting->identity contiguity->rp8"),
    ("2  local256+str8", "local256+str8", 1024, 128, 128, (None, 2),
     {"identity": (36, 8), "rp2": (34, 15)}, "counting->rp2 contiguity->identity"),
    ("3  local256+str8", "local256+str8", 2048, 128, 32, (None, 2, 4),
     {"rp2": (408, 40), "rp4": (424, 34)}, "counting->rp2 1.04x  makespan->rp4 1.18x"),
]


class Best(_NT("Best", "ms sd warps stages")):
    __slots__ = ()

    @property
    def cfg(self):
        return f"w{self.warps}/s{self.stages}"


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def _counts(M, BQ, A):
    t = M.reshape(M.shape[0] // BQ, BQ, M.shape[1] // A, A).any(axis=(1, 3))
    runs = sum(0 if (i := np.flatnonzero(r)).size == 0
               else 1 + int((np.diff(i) > 1).sum()) for r in t)
    return int(t.sum()), int(t.sum(axis=1).max()), runs


def main():
    torch.manual_seed(0)
    for label, name, N, BQ, A, folds, expect, note in CELLS:
        print(f"\n=== CELL {label}  N={N} {BQ}x{A}   {note}")
        print(f"    programs {(N//BQ)*BH}, ~{(N//BQ)*BH/108:.2f} waves on 108 SMs")
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        M0 = np.stack([m.row_cols(i, N) for i in range(N)])
        q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
                   for _ in range(3))

        res, _PAIRED = {}, {}
        _thunks = {}
        for s in folds:
            tag = "identity" if s is None else f"rp{s}"
            p_np = (permute.identity_perm(N) if s is None
                    else permute.residue_perm(N, s))
            perm = torch.from_numpy(p_np).cuda()
            Mv = M0[p_np][:, p_np]
            tot, mx, runs = _counts(Mv, BQ, A)
            if tag in expect and (tot, runs if BQ == 128 and A == 128 else mx) != expect[tag]:
                print(f"    !! {tag}: counts {(tot, mx, runs)} do not match the "
                      f"recorded {expect[tag]} -- the cell has drifted, STOP")
                return 1
            bi = blockindex.build(_Dense(Mv), N, BQ, A)
            idx = blockindex.to_cuda(bi)
            qp, kp, vp = (permute.apply_perm(x, perm) for x in (q, k, v))
            # NAMED, not positional. This was a 3-tuple (ms, sd, "w1/s3") and
            # the paired-timing edit indexed best[3] for the stage count -- an
            # IndexError that syntax-checked clean and killed both exp8 jobs on
            # the GPU. Fields cannot be miscounted.
            best = Best(float("inf"), 0.0, None, None)
            for w, st in LAUNCH:
                try:
                    ms, sd = bench.time_ms(lambda: block_sparse_attention(
                        qp, kp, vp, *idx, kind, p0, p1, p2, BQ, A,
                        perm_q=perm.int(), perm_kv=perm.int(),
                        num_warps=w, num_stages=st), warmup=25, reps=200)
                except Exception:
                    continue
                if ms < best.ms:
                    best = Best(ms, sd, w, st)
            res[tag] = (tot, mx, runs, best.ms, best.sd, best.cfg)
            bw, bst = best.warps, best.stages
            if bw is not None:
                _thunks[tag] = lambda qp=qp, kp=kp, vp=vp, idx=idx, perm=perm, \
                                      BQ=BQ, A=A, bw=bw, bst=bst: \
                    block_sparse_attention(qp, kp, vp, *idx, kind, p0, p1, p2,
                                           BQ, A, perm_q=perm.int(),
                                           perm_kv=perm.int(), num_warps=bw,
                                           num_stages=bst)

        for tag, (tot, mx, runs, ms, sd, cfg) in res.items():
            print(f"    {tag:<10} tiles {tot:>5}  max {mx:>4}  runs {runs:>4}   "
                  f"{ms:.4f} +- {sd:.4f} ms  [{cfg}]"
                  + _warn_if_on_boundary(cfg))
        # PAIRED re-measurement of the two comparands, INTERLEAVED. The loop
        # above times each variant fully before starting the next, so any drift
        # over that window is confounded with the A-vs-B difference -- and cell 2's
        # margin is 5.9% against a 2% CV. Alternating cancels drift to first
        # order; the paired RATIO is the statistic to trust, not the medians.
        tags = list(res)[:2]
        if len(tags) == 2 and all(t in _thunks for t in tags):
            _PAIRED[label] = (_thunks[tags[0]], _thunks[tags[1]])
        if _PAIRED.get(label) is not None:
            fa_, fb_ = _PAIRED[label]
            ma, mb, rat, rsd = bench.paired_time(fa_, fb_, warmup=15, reps=120)
            print(f"    PAIRED (interleaved)  {tags[0]} {ma:.4f}  {tags[1]} {mb:.4f}"
                  f"   ratio {rat:.4f} +- {rsd:.4f}")
            res[tags[0]] = res[tags[0]][:3] + (ma, rsd * ma) + res[tags[0]][5:]
            res[tags[1]] = res[tags[1]][:3] + (mb, rsd * mb) + res[tags[1]][5:]

        cv = {t: r[4] / r[3] for t, r in res.items() if r[3]}
        print(f"    coefficient of variation: "
              + "  ".join(f"{t} {c*100:.1f}%" for t, c in cv.items()))

        if label.startswith("3"):
            # DIRECTION only, boundary 1.0, symmetric band at the measured spread.
            # No midpoint: neither point prediction is trusted, so no rule derived
            # from both of them is either.
            t_a, t_b = res["rp2"][3], res["rp4"][3]
            band = 2 * (res["rp2"][4] + res["rp4"][4]) / t_a
            ratio = t_b / t_a
            print(f"    measured rp4/rp2 = {ratio:.4f}   "
                  f"+-2 sigma band around 1.0 = [{1-band:.4f}, {1+band:.4f}]")
            if abs(ratio - 1.0) <= band:
                print("    VERDICT: COULD NOT RESOLVE -- inside the spread of 1.0.")
                print("             A quieter configuration, not more reps.")
            elif ratio < 1.0:
                print("    VERDICT: WAVE/MAKESPAN side. Counting picked rp2; hardware")
                print("             picked rp4, which computes MORE elements.")
            else:
                print("    VERDICT: COUNTING side. The makespan account is wrong outright")
                print("             -- rp4 has the shorter longest job and still lost.")
        else:
            (t1, r1), (t2, r2) = list(res.items())[:2]
            d = abs(r1[3] - r2[3])
            noise = 2 * (r1[4] + r2[4])
            win = t1 if r1[3] < r2[3] else t2
            if d < noise:
                print(f"    VERDICT: COULD NOT RESOLVE -- difference {d:.4f} ms is inside"
                      f" +-2 sigma ({noise:.4f} ms). Honest null, not a negative.")
            else:
                print(f"    VERDICT: {win} wins by "
                      f"{max(r1[3], r2[3])/min(r1[3], r2[3]):.3f}x "
                      f"(diff {d:.4f} ms vs +-2 sigma {noise:.4f} ms)")
    return 0


if __name__ == "__main__":
    # `python exp8_three_cells.py 3` runs cell 3 only. Under queue uncertainty it
    # is the single most valuable thing in the whole set: the ONLY parameter-free
    # falsifier -- it returns a direction, and if rp2 wins the wave account is
    # dead outright with no coefficient to retreat behind. Everything else,
    # exp6 included, returns a number some account can absorb.
    if len(sys.argv) > 1:
        want = set(sys.argv[1:])
        CELLS = [c for c in CELLS if c[0].split()[0] in want]
        print(f"running only cell(s) {sorted(want)}\n")
    raise SystemExit(main())
