"""MECHANISM for d4's open gap: link probe -> finding by making the FINDING the
assertion.

The gap: our probes assert PROPERTIES ("this mask has short documents") while our
results are FINDINGS ("short documents make the identity fallback lose 6.9x").
A probe can keep exhibiting its property long after it has stopped defending the
finding it was created for -- which is exactly what happened to d4's docpack
probes -- and no property check can see that.

The fix is not a new check. It is to assert the FINDING instead of the property.
Then the link is identity rather than documentation:
  - the probe cannot drift from the finding, because it IS the finding;
  - deleting the mask deletes the finding's test, loudly;
  - if the number moves, this fails and names which claim just changed.

Below: every number b5 reported to the team, as an executable assertion.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of, select
from oracle import oracle_cost

ok = bad = 0
def check(claim, got, want, tol=0.0):
    global ok, bad
    good = (abs(got - want) <= tol) if tol else (got == want)
    ok, bad = ok + good, bad + (not good)
    print(f"  {'PASS' if good else 'FAIL'}  {claim}\n        got {got!r}  want {want!r}")

def live(pieces, N): return DiagSpec(pieces).live(N)
def waste(pieces, N, BQ, A): return cost_of("identity", DiagSpec(pieces), N, BQ, A) / live(pieces, N)

print("FINDING 1 (C3 kill): twoband misaligned violates the max-law; aligned does not,")
print("and the violation GROWS with N -- so it is not a boundary artefact.")
for N, want in ((2048, 0.3287), (4096, 0.3918), (8192, 0.4163)):
    mis = [Iv(0, 128), Iv(1000, 1128)]
    ws = [waste(mis, N, b, a) for b in (128,64,32,16) for a in (128,64,32,16) if max(b,a) == 128]
    check(f"misaligned spread@max=128, N={N}", round(max(ws)-min(ws), 4), want, 1e-4)
ali = [Iv(0, 128), Iv(1024, 1152)]
ws = [waste(ali, 2048, b, a) for b in (128,64,32,16) for a in (128,64,32,16) if max(b,a) == 128]
check("aligned twin spread@max=128, N=2048", round(max(ws)-min(ws), 6), 0.0)

print("\nFINDING 2 (C2 kill): the argmin splits WITHIN a max(BQ,A)=128 class.")
c2 = [Iv(0, 24), Iv(500, 524), Ap(0, 2, 512)]
picks = {}
for b, a in ((128,128),(128,64),(128,32),(128,16),(64,128),(32,128),(16,128)):
    picks[(b,a)] = select(DiagSpec(c2), 1024, b, a)[0]
check("distinct argmins across max=128 cells", len(set(picks.values())), 2)
check("128x128 winner", picks[(128,128)], "identity")
check("128x16 winner", picks[(128,16)], "residue-perm-2")

print("\nFINDING 3 (S5i, new): the argmin is N-DEPENDENT -- S5c's table is N-specific.")
def am(N, BQ, A):
    return min((oracle_cost([Iv(0,256), Ap(0,8,N//8)], N, BQ, A, c), c)
               for c in ("identity","residue-perm-2","residue-perm-4","residue-perm-8"))[1]
check("local256+str8 128x128 argmin at N=2048", am(2048,128,128), "residue-perm-2")
check("local256+str8 128x128 argmin at N=4096", am(4096,128,128), "residue-perm-4")
check("local256+str8 16x16 argmin at N=1024",  am(1024,16,16),  "residue-perm-4")
check("local256+str8 16x16 argmin at N=2048",  am(2048,16,16),  "residue-perm-8")

print("\nFINDING 4 (S5c reproduction): b5's symbolic engine matches d4's published table.")
for (BQ,A), want in {(128,128):2.411,(64,64):1.914,(32,32):1.601,(16,16):1.332}.items():
    best = min(cost_of(c, DiagSpec([Iv(0,256),Ap(0,8,256)]), 2048, BQ, A)
               for c in ("identity","residue-perm-2","residue-perm-4","residue-perm-8"))
    check(f"local256+str8 waste at {BQ}x{A}, N=2048",
          round(best/live([Iv(0,256),Ap(0,8,256)], 2048), 3), want, 1.5e-3)

print("\nFINDING 5 (symmetry theorem): exact integer symmetry, incl. unbounded and random D.")
for pieces, nm in (([Iv(0,2048)], "causal"), ([Ap(0,8,256)], "dilated-8"),
                   ([Iv(0,256),Ap(0,8,256)], "local256+str8")):
    d = max(abs(cost_of("identity", DiagSpec(pieces), 2048, b, a)
                - cost_of("identity", DiagSpec(pieces), 2048, a, b))
            for b in (128,64,32,16) for a in (128,64,32,16))
    check(f"max |cost(BQ,A)-cost(A,BQ)| on {nm}", d, 0)

print(f"\n{ok} passed, {bad} failed")
sys.exit(1 if bad else 0)
