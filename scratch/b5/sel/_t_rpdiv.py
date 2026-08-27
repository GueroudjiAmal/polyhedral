"""Is my rp_s divisibility guard protecting the MATHEMATICS or my ALGORITHM?
And could my own declined-but-costable metric have caught it?"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of
import oracle

print("1. Is sorting by (i mod s, i div s) a bijection when s does not divide N?")
for N, s in ((1024,3),(1024,5),(2048,3),(2048,5),(1000,7)):
    order = np.argsort(np.arange(N) % s * N + np.arange(N)//s, kind="stable")
    sizes = [int(((np.arange(N) % s) == r).sum()) for r in range(s)]
    print(f"   N={N} s={s}: bijection={sorted(order.tolist())==list(range(N))}"
          f"  class sizes {sizes[:4]}{'...' if s>4 else ''}")
print("   -> yes. 2f is right: the permutation is well defined for any s.")

print("\n2. WHERE my guard actually lives:")
print("   xform.cost_residue_perm returns None on N % s because my BLOCK")
print("   DECOMPOSITION assumes s equal-sized residue blocks of size B=N/s.")
print("   With ragged classes (sizes differing by 1) that decomposition breaks.")
print("   So the guard protects MY ALGORITHM, not the transform. 2f is right.")

print("\n3. COULD MY declined-but-costable METRIC HAVE CAUGHT IT?")
print("   That metric counts: engine returns None AND oracle returns a cost.")
print("   My oracle.apply_xform has the SAME guard:")
import inspect
src = inspect.getsource(oracle.apply_xform)
print("     " + [l.strip() for l in src.splitlines() if "n % s" in l][0])
print("   so BOTH decline and the pair is never counted. The metric read 0")
print("   because oracle and engine share the assumption -- exactly the")
print("   'validated only against another implementation of itself' failure")
print("   I warned d4 about this morning, in my own harness.")

print("\n4. HONEST NUMBER: patch the oracle to allow any s, then re-measure.")
def apply_any(M, name):
    if not name.startswith("residue-perm-"):
        return oracle.apply_xform(M, name)
    s = int(name.split("-")[-1]); n = M.shape[0]
    order = np.argsort(np.arange(n) % s * n + np.arange(n)//s, kind="stable")
    return M[order][:, order]

MASKS = {"local256+str8": lambda N: [Iv(0,256), Ap(0,8,N//8)],
         "window-128":    lambda N: [Iv(0,128)],
         "dilated-8":     lambda N: [Ap(0,8,N//8)]}
CANDS = [f"residue-perm-{s}" for s in (2,3,4,5,6,8,12,16,32)]
dec = tot = 0
for N in (1024, 2048):
    for nm, mk in MASKS.items():
        M0 = oracle.dense(mk(N), N)
        for BQ, A in ((128,128),(128,16),(32,32),(16,16)):
            for c in CANDS:
                got = cost_of(c, DiagSpec(mk(N)), N, BQ, A)
                exp = oracle.tiles_cost(apply_any(M0, c), BQ, A)
                tot += 1
                if got is None and exp is not None:
                    dec += 1
print(f"   declined-but-costable, corrected oracle: {dec}/{tot}"
      f"  ({dec/tot*100:.0f}% of candidate evaluations)")
print("   previously reported: 0. The 0 was an artefact of the shared guard.")
