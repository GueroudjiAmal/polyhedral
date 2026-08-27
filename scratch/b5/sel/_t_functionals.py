"""Replacing my broken enumeration with a CHARACTERISATION, and testing it.

My corollary enumerated hardware terms and missed one. Enumerating again is the
same move. The closed statement instead:

  A kernel's runtime is a FUNCTIONAL OF THE TILE SET T(t) that the transform
  induces. Element count is ONE scalar summary of that set -- its cardinality.
  The theorem says cardinality-preserving corrections are inert. Therefore any
  OTHER summary of T that differs by transform is potentially decision-relevant.

That is checkable rather than open-ended: enumerate the SUMMARIES of a set, not
the phenomena. Five natural ones, and which are known to bite:

  S1 cardinality              |T|                          -> elements (INERT, theorem)
  S2 distribution over rows   max tiles per row-block       -> imbalance (2f)
  S3 per-tile footprint       distinct kv rows per tile     -> class A/B (b5)
  S4 spatial locality         column-tile runs per row-block-> coalescing (untested)
  S5 temporal reuse           distinct column-tiles overall -> L2 reuse (untested)

S4 and S5 are tested here for the first time.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summaries(M, BQ, A):
    nq, nk = M.shape
    P = np.pad(M, ((0, (-nq) % BQ), (0, (-nk) % A)))
    t = P.reshape(-1, BQ, P.shape[1] // A, A).any(axis=(1, 3))
    per = t.sum(axis=1)
    # S4: number of maximal runs of consecutive live column-tiles, per row-block,
    #     summed -- fewer runs = more coalesced
    runs = 0
    for row in t:
        r = np.diff(np.concatenate(([0], row.astype(int), [0])))
        runs += int((r == 1).sum())
    return {"S1 cardinality": int(t.sum()),
            "S2 max/row-block": int(per.max()),
            "S4 column runs": runs,
            "S5 distinct cols": int(t.any(axis=0).sum())}

CANDS = ["identity", "residue-perm-2", "residue-perm-4", "residue-perm-8"]
MASKS = {"local256+str8": lambda N: [Iv(0,256), Ap(0,8,N//8)],
         "local128+str4": lambda N: [Iv(0,128), Ap(0,4,N//4)],
         "dilated-8":     lambda N: [Ap(0,8,N//8)],
         "window-128":    lambda N: [Iv(0,128)],
         "twoband-mis":   lambda N: [Iv(0,128), Iv(1000,1128)]}
TIL = [(128,128),(128,32),(128,16),(64,64),(32,32),(16,16)]

keys = ["S2 max/row-block", "S4 column runs", "S5 distinct cols"]
dis = {k: 0 for k in keys}; tot = 0
examples = {k: None for k in keys}
for N in (1024, 2048):
    for nm, mk in MASKS.items():
        M0 = dense(mk(N), N)
        for BQ, A in TIL:
            S = {}
            for c in CANDS:
                Mt = apply_xform(M0, c)
                if Mt is not None:
                    S[c] = summaries(Mt, BQ, A)
            if len(S) < 2: continue
            tot += 1
            base = min(S, key=lambda c: S[c]["S1 cardinality"])
            for k in keys:
                alt = min(S, key=lambda c: S[c][k])
                if alt != base:
                    dis[k] += 1
                    if examples[k] is None:
                        examples[k] = (nm, N, BQ, A, base, alt,
                                       S[base][k], S[alt][k])
print(f"cells: {tot}\n")
print(f"{'summary':<20}{'argmin differs from cardinality':>34}")
for k in keys:
    print(f"{k:<20}{f'{dis[k]}/{tot}':>34}")
print("\nfirst disagreement for each:")
for k in keys:
    e = examples[k]
    if e:
        print(f"  {k:<20}{e[0]} N={e[1]} {e[2]}x{e[3]}: "
              f"count->{e[4]}, {k}->{e[5]} ({e[6]} vs {e[7]})".replace("residue-perm-","rp"))
    else:
        print(f"  {k:<20}none -- this summary never disagrees with cardinality")
