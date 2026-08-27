"""Two corrections to my own S1-S5 table, both from 2f.
  1. STRICT vs LOOSE: a tie-broken argmin is not a disagreement. I applied this
     exact check to the class A/B tie population this morning and failed to apply
     it here.
  2. S4 per TILE SHAPE, not pooled: a 'run' is 16 columns at A=16 and 128 at
     A=128, so pooling across tile shapes hides whatever structure there is.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summaries(M, BQ, A):
    nq, nk = M.shape
    P = np.pad(M, ((0, (-nq) % BQ), (0, (-nk) % A)))
    t = P.reshape(-1, BQ, P.shape[1] // A, A).any(axis=(1, 3))
    runs = 0
    for row in t:
        r = np.diff(np.concatenate(([0], row.astype(int), [0])))
        runs += int((r == 1).sum())
    return {"S1": int(t.sum()), "S2": int(t.sum(axis=1).max()),
            "S4": runs, "S5": int(t.any(axis=0).sum())}

CANDS = ["identity", "residue-perm-2", "residue-perm-4", "residue-perm-8"]
UNION = {"local256+str8": lambda N: [Iv(0,256), Ap(0,8,N//8)],
         "local128+str4": lambda N: [Iv(0,128), Ap(0,4,N//4)],
         "local64+str2":  lambda N: [Iv(0,64),  Ap(0,2,N//2)]}
OTHER = {"dilated-8": lambda N: [Ap(0,8,N//8)], "dilated-4": lambda N: [Ap(0,4,N//4)],
         "window-128": lambda N: [Iv(0,128)], "twoband-mis": lambda N: [Iv(0,128), Iv(1000,1128)]}
TIL = [(128,128),(128,64),(128,32),(128,16),(64,64),(64,16),(32,32),(16,16)]

def scan(masks):
    out = {}
    for N in (1024, 2048):
        for nm, mk in masks.items():
            M0 = dense(mk(N), N)
            for BQ, A in TIL:
                S = {c: summaries(m, BQ, A) for c in CANDS
                     if (m := apply_xform(M0, c)) is not None}
                if len(S) < 2: continue
                b = min(S, key=lambda c: S[c]["S1"])
                for k in ("S2", "S4", "S5"):
                    a = min(S, key=lambda c: S[c][k])
                    loose = (a != b)
                    strict = loose and S[a][k] != S[b][k]
                    d = out.setdefault((k, BQ, A), [0, 0, 0])
                    d[0] += 1; d[1] += loose; d[2] += strict
    return out

print("=== STRICT vs LOOSE, union masks, pooled ===")
u = scan(UNION)
for k in ("S2", "S4", "S5"):
    tot = sum(v[0] for (kk, _, _), v in u.items() if kk == k)
    lo = sum(v[1] for (kk, _, _), v in u.items() if kk == k)
    st = sum(v[2] for (kk, _, _), v in u.items() if kk == k)
    print(f"  {k}   cells {tot:>3}   loose {lo:>3}   STRICT {st:>3}"
          + ("   <- entirely tie artefact" if lo and not st else ""))

print("\n=== non-union masks (lattice / simple), pooled ===")
o = scan(OTHER)
for k in ("S2", "S4", "S5"):
    tot = sum(v[0] for (kk, _, _), v in o.items() if kk == k)
    st = sum(v[2] for (kk, _, _), v in o.items() if kk == k)
    print(f"  {k}   cells {tot:>3}   STRICT {st:>3}")

print("\n=== S4 per TILE SHAPE (strict), union masks -- 2f's point, untested by either of us ===")
print(f"{'tile':>10}{'cells':>7}{'strict':>8}{'rate':>8}")
for BQ, A in TIL:
    v = u.get(("S4", BQ, A))
    if v:
        print(f"{f'{BQ}x{A}':>10}{v[0]:>7}{v[2]:>8}{v[2]/v[0]*100:>7.0f}%")
