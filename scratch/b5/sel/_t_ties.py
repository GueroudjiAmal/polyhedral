"""Does my engine have d4's class-A/class-B tie, and which way does it resolve?"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of, select

CLASS_B = lambda n: n == "shear" or n.startswith("stridefold")
MASKS = [("dilated-8", lambda N: [Ap(0, 8, N // 8)]),
         ("dilated-4", lambda N: [Ap(0, 4, N // 4)]),
         ("dilated-2", lambda N: [Ap(0, 2, N // 2)]),
         ("local256+str8", lambda N: [Iv(0, 256), Ap(0, 8, N // 8)]),
         ("window-128", lambda N: [Iv(0, 128)]),
         ("causal", lambda N: [Iv(0, N)]),
         ("twoband-mis", lambda N: [Iv(0, 128), Iv(1000, 1128)])]
TIL = [128, 64, 32, 16]
tot = ties = shipped_B = 0
examples = []
for N in (1024, 2048, 4096):
    for nm, mk in MASKS:
        D = DiagSpec(mk(N))
        for BQ in TIL:
            for A in TIL:
                costs = {}
                for c in CANDIDATES:
                    v = cost_of(c, D, N, BQ, A)
                    if v is not None:
                        costs[c] = v
                if not costs:
                    continue
                tot += 1
                best = min(costs.values())
                at_best = [c for c in costs if costs[c] == best]
                if any(CLASS_B(c) for c in at_best) and any(not CLASS_B(c) for c in at_best):
                    ties += 1
                    pick, _ = select(D, N, BQ, A)
                    if CLASS_B(pick):
                        shipped_B += 1
                        examples.append((nm, N, BQ, A, pick, at_best))
print(f"costed instances            {tot}")
print(f"class-A / class-B TIES      {ties}  ({ties/tot*100:.1f}%)")
print(f"of those, shipped class B   {shipped_B}")
if examples:
    for e in examples[:6]:
        print(f"   {e[0]} N={e[1]} {e[2]}x{e[3]} -> {e[4]}  (tied: {e[5]})")
else:
    print("   -> my select() breaks ties toward class A by construction "
          "(xform.py: key = (cost, 0 if identity/residue else 1, name))")
