"""From two empirical verifications to a theorem, and a tightness check.

THEOREM (transform-independent corrections are argmin-inert).
Let a time model have the form
        time(t, BQ, A) = g( elements(t, BQ, A), BQ, A )
i.e. the transform t enters ONLY through the element count, and let g be
STRICTLY INCREASING in its first argument. Then for every (BQ, A)
        argmin_t time(t, BQ, A) == argmin_t elements(t, BQ, A).
Proof: at fixed (BQ,A), g(., BQ, A) is a strictly increasing function of one
variable, and a strictly increasing map preserves order, hence preserves argmin.

This covers BOTH corrections proposed so far as special cases:
  2f  multiplicative   g = p(BQ,A) * e                    increasing in e
  d4  additive fixed   g = nprog*F + e*V/(BQ*A)           increasing in e
and every other member of the family, including non-linear ones nobody has
proposed yet -- occupancy curves, saturating throughput, roofline knees.

So it is not that 2f picked the wrong functional form. NO correction of this
shape can move a transform decision, ever.

COROLLARY. A correction can change a transform decision only if it depends on t
through something OTHER than the element count. In this taxonomy the only such
quantity is MEMORY TRAFFIC (kv rows per tile), which is transform-dependent by
construction and is exactly the class A / class B axis.

Below: numerical confirmation on a range of monotone g including strongly
non-linear ones, plus a TIGHTNESS check that a NON-monotone g does flip argmins
(so 'strictly increasing' is load-bearing, not decoration).
"""
import sys, math; sys.path.insert(0, ".")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of

MASKS = [("dilated-2", lambda N: [Ap(0,2,N//2)]), ("dilated-8", lambda N: [Ap(0,8,N//8)]),
         ("local256+str8", lambda N: [Iv(0,256), Ap(0,8,N//8)]),
         ("window-128", lambda N: [Iv(0,128)]), ("twoband-mis", lambda N: [Iv(0,128), Iv(1000,1128)])]
TIL = [128, 64, 32, 16]

GS = {
    "identity                e":            lambda e,b,a: e,
    "multiplicative (2f)     p(b,a)*e":     lambda e,b,a: (1.0 + 64.0/min(b,a)) * e,
    "additive fixed (d4)     F*nprog+e*V":  lambda e,b,a: (4096//b)*8*5e3 + e*(1.0/(b*a)),
    "sqrt (saturating)       sqrt(e)":      lambda e,b,a: math.sqrt(e),
    "log (extreme concave)   log1p(e)":     lambda e,b,a: math.log1p(e),
    "roofline knee           min(e,K)+e/8": lambda e,b,a: min(e, 2e6) + e/8,
    "occupancy-shaped        e*(1+8/ b)":   lambda e,b,a: e * (1.0 + 8.0/b),
}
NONMONO = {"NON-monotone (tightness) -e": lambda e,b,a: -e}

def run(gs, label):
    print(f"\n{label}")
    for gname, g in gs.items():
        flips = tot = 0
        for N in (2048, 4096):
            for nm, mk in MASKS:
                D = DiagSpec(mk(N))
                for b in TIL:
                    for a in TIL:
                        costs = {c: v for c in CANDIDATES
                                 if (v := cost_of(c, D, N, b, a)) is not None}
                        if len(costs) < 2:
                            continue
                        base = min(costs, key=lambda c: costs[c])
                        corr = min(costs, key=lambda c: g(costs[c], b, a))
                        tot += 1
                        flips += (base != corr)
        print(f"  {gname:<34} argmin changes {flips:>4}/{tot}")

run(GS, "MONOTONE g -- theorem predicts 0 changes for every one:")
run(NONMONO, "NON-monotone g -- theorem makes no promise; tightness check:")
