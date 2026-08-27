"""Verify 2f's list-scheduling bound for experiment 3, from my own counts."""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

SMS, H = 108, 8
def stats(M, BQ, A):
    nq, nk = M.shape
    P = np.pad(M, ((0,(-nq)%BQ),(0,(-nk)%A)))
    t = P.reshape(-1,BQ,P.shape[1]//A,A).any(axis=(1,3))
    per = t.sum(axis=1)
    return int(t.sum()), int(per.max())

print("experiment 3: local256+str8, 128x32, list-scheduling bound = max(longest job, total/SMs)")
print(f"{'N':>6}{'cand':>10}{'tiles/head':>12}{'max job':>9}{'progs':>7}{'waves':>7}"
      f"{'work/SM':>9}{'bound':>7}")
for N in (1024, 2048):
    M0 = dense([Iv(0,256), Ap(0,8,N//8)], N)
    nprog = (N//128) * H
    rows = {}
    for c in ("identity","residue-perm-2","residue-perm-4","residue-perm-8"):
        m = apply_xform(M0, c)
        if m is None: continue
        tot, mx = stats(m, 128, 32)
        work = tot * H / SMS
        bound = max(mx, work)
        rows[c] = (tot, mx, bound)
        print(f"{N:>6}{c.replace('residue-perm-','rp'):>10}{tot:>12}{mx:>9}{nprog:>7}"
              f"{nprog/SMS:>7.2f}{work:>9.1f}{bound:>7.1f}")
    by_count = min(rows, key=lambda c: rows[c][0])
    by_bound = min(rows, key=lambda c: rows[c][2])
    print(f"       counting -> {by_count.replace('residue-perm-','rp')}"
          f"   list-sched -> {by_bound.replace('residue-perm-','rp')}"
          f"   margin {max(rows[by_count][2], rows[by_bound][2])/min(rows[by_count][2], rows[by_bound][2]):.3f}x"
          + ("   AGREE" if by_count == by_bound else "   DISAGREE"))
    print()

print("""PRECISION REQUIRED, since the two predictions are close and opposed:
  wave/list-sched says rp4 faster than rp2 by ~1.18x
  counting says      rp2 faster than rp4 by ~1.04x
  total spread between the two hypotheses = 1.18 * 1.04 = 1.23x
So the measurement must resolve rp4-vs-rp2 to well inside 23%. At 5% noise the
hypotheses are 4-5 sigma apart and separable; at 15% they are not. 2f's call for
best-of over a launch-config sweep with enough reps to separate medians is right,
and the number d4 needs in advance is 23%, not just 'the margin is tight'.

CAVEAT ON THE BOUND: max(longest job, total/SMs) is a LOWER bound on makespan.
Real list scheduling can reach (2 - 1/m) times optimal, and it is a bound on the
SCHEDULE, not on the kernel -- it ignores per-tile cost variation, tail effects
within a program, and memory. So 1.18x is a ratio of lower bounds, not a
predicted speedup. The DIRECTION is the parameter-free part; the magnitude is
not, and I would put only the direction on the record as the falsifier.""")
