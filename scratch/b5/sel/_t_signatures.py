"""Candidate BEHAVIOURAL signatures for d4's two remaining shape checks.
Proposing an unverified check would be the same error we have been cataloguing,
so both are measured here before being suggested."""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of as dcost
import general as G

TIL = [(128,128),(128,32),(64,64),(32,32),(16,16)]

print("=== randD-600: proposed signature = NO transform meaningfully beats identity ===")
print("   (if a random displacement set DID admit a big win it is not unstructured)")
print(f"{'mask':<22}{'N':>6}{'tile':>9}{'identity':>13}{'best rp':>13}{'gain':>7}")
rng = np.random.default_rng(3)
for N in (1024, 2048):
    offs = sorted(rng.choice(N, min(600, N), replace=False).tolist())
    D = DiagSpec([Iv(o, o+1) for o in offs])
    struct = DiagSpec([Iv(0,256), Ap(0,8,N//8)])          # control: structured
    for nm, spec in (("randD-600", D), ("local256+str8 (ctl)", struct)):
        for BQ, A in ((128,128),(16,16)):
            i = dcost("identity", spec, N, BQ, A)
            best = min(v for c in ("residue-perm-2","residue-perm-4","residue-perm-8",
                                   "residue-perm-16","residue-perm-32")
                       if (v := dcost(c, spec, N, BQ, A)) is not None)
            print(f"{nm:<22}{N:>6}{f'{BQ}x{A}':>9}{i:>13,}{best:>13,}{i/best:>7.2f}x")

print("\n=== docpack mixed 2/895: proposed signature = behaves like NEITHER uniform twin ===")
def ddoc(b, N):
    bb = np.array(list(b)+[N]); d = np.searchsorted(bb, np.arange(N), side="right")-1
    q = np.arange(N)[:,None]; kv = np.arange(N)[None,:]
    return (kv<=q)&(d[:,None]==d[None,:])
from oracle import tiles_cost, apply_xform
N = 1024
SETS = {"mixed 2/895": [0,2,897],
        "uniform tiny (2)": list(range(0,N,2)),
        "uniform long (895)": [0,895]}
print(f"{'variant':<22}{'tile':>9}{'argmin':>16}{'identity regret':>17}")
res = {}
for nm, b in SETS.items():
    M0 = ddoc(b, N); sp = G.DocPack(b, nm)
    for BQ, A in ((128,128),(32,32)):
        costs = {c: tiles_cost(m, BQ, A) for c in G.CANDIDATES
                 if (m := apply_xform(M0, c)) is not None}
        best = min(costs.values()); am = min(costs, key=costs.get)
        reg = costs["identity"]/best
        res[(nm,BQ,A)] = (am, reg)
        print(f"{nm:<22}{f'{BQ}x{A}':>9}{am:>16}{reg:>17.3f}")
print("\n  signature fires if mixed's (argmin, identity-regret) differs from BOTH twins:")
for BQ, A in ((128,128),(32,32)):
    m_, t_, l_ = res[("mixed 2/895",BQ,A)], res[("uniform tiny (2)",BQ,A)], res[("uniform long (895)",BQ,A)]
    diff_t = (m_[0] != t_[0]) or abs(m_[1]-t_[1]) > 0.05
    diff_l = (m_[0] != l_[0]) or abs(m_[1]-l_[1]) > 0.05
    print(f"    {BQ}x{A}: differs from tiny={diff_t}, from long={diff_l}"
          f"  -> {'FIRES' if diff_t and diff_l else 'does NOT fire'}")
