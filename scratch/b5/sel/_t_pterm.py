"""Can a hardware penalty term p change a TRANSFORM selection?

Claim to test: if p depends only on (BQ, A), then at a FIXED tile shape it
multiplies every candidate equally, so argmin_t [cost(t,BQ,A) * p(BQ,A)] ==
argmin_t cost(t,BQ,A). p cannot move a transform argmin -- only a TILE-SHAPE
choice. It can only move a transform argmin if p is TRANSFORM-DEPENDENT, which it
is for class B: shear/stridefold change kv rows per tile, so they carry a traffic
penalty class A does not.

So the question "does the hardware term change any selection decision" splits:
  transform at fixed tile shape  -> only a CLASS-DEPENDENT p can change it
  tile shape                     -> p governs it entirely
Below: how sensitive is the transform argmin to a class-B penalty factor?
"""
import sys; sys.path.insert(0, ".")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of

CLASS_B = lambda n: n == "shear" or n.startswith("stridefold")
MASKS = [("dilated-2", lambda N: [Ap(0,2,N//2)]), ("dilated-4", lambda N: [Ap(0,4,N//4)]),
         ("dilated-8", lambda N: [Ap(0,8,N//8)]),
         ("local256+str8", lambda N: [Iv(0,256), Ap(0,8,N//8)]),
         ("window-128", lambda N: [Iv(0,128)]), ("causal", lambda N: [Iv(0,N)]),
         ("twoband-mis", lambda N: [Iv(0,128), Iv(1000,1128)])]
TIL = [128, 64, 32, 16]

tot = flips = already_A = 0
thresholds = []
for N in (2048, 4096):
    for nm, mk in MASKS:
        D = DiagSpec(mk(N))
        for BQ in TIL:
            for A in TIL:
                costs = {c: v for c in CANDIDATES
                         if (v := cost_of(c, D, N, BQ, A)) is not None}
                if len(costs) < 2:
                    continue
                tot += 1
                bA = min((v for c, v in costs.items() if not CLASS_B(c)), default=None)
                bB = min((v for c, v in costs.items() if CLASS_B(c)), default=None)
                if bA is None or bB is None:
                    continue
                if bB < bA:
                    # class B currently wins on elements; what penalty flips it to A?
                    thresholds.append(bA / bB)
                    flips += 1
                else:
                    already_A += 1

print(f"instances with both classes costable   {flips + already_A}")
print(f"  class A already wins or ties          {already_A}")
print(f"  class B wins on ELEMENTS              {flips}")
if thresholds:
    thresholds.sort()
    print(f"\nclass-B penalty factor needed to flip those {flips} to class A:")
    print(f"  min {thresholds[0]:.3f}   median {thresholds[len(thresholds)//2]:.3f}"
          f"   max {thresholds[-1]:.3f}")
    for t in (1.05, 1.2, 1.5, 1.94, 8.5):
        n = sum(1 for x in thresholds if x <= t)
        print(f"  a penalty of {t:>4.2f}x flips {n:>3}/{flips} of them")
print("""
NOTE the two reference penalties already in NOTES:
  1.94x  -- shear on window-128 raises kv rows/tile 16 -> 31 (sec 4)
  8.5x   -- stridefold-8 at 16x16 touches 136 kv rows vs 16 (d4's tie table)
A class-B traffic term of that size flips essentially all of them.""")
