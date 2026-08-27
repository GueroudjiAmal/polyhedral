"""Which masks produce the 4 non-union S4 disagreements? Either my prediction
('vanishes on pure lattices and simple masks') is falsified, or twoband-mis is
misclassified -- it is a union of two intervals, which is a union mask."""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summ(M, BQ, A):
    nq, nk = M.shape
    P = np.pad(M, ((0,(-nq)%BQ),(0,(-nk)%A)))
    t = P.reshape(-1,BQ,P.shape[1]//A,A).any(axis=(1,3))
    runs = sum(int((np.diff(np.concatenate(([0],r.astype(int),[0])))==1).sum()) for r in t)
    return int(t.sum()), runs

CANDS=["identity","residue-perm-2","residue-perm-4","residue-perm-8"]
OTHER={"dilated-8":lambda N:[Ap(0,8,N//8)], "dilated-4":lambda N:[Ap(0,4,N//4)],
       "window-128":lambda N:[Iv(0,128)], "twoband-mis":lambda N:[Iv(0,128),Iv(1000,1128)]}
TIL=[(128,128),(128,64),(128,32),(128,16),(64,64),(64,16),(32,32),(16,16)]
hits={}
for N in (1024,2048):
    for nm,mk in OTHER.items():
        M0=dense(mk(N),N)
        for BQ,A in TIL:
            S={c:summ(m,BQ,A) for c in CANDS if (m:=apply_xform(M0,c)) is not None}
            if len(S)<2: continue
            b=min(S,key=lambda c:S[c][0]); a=min(S,key=lambda c:S[c][1])
            if a!=b and S[a][1]!=S[b][1]:
                hits.setdefault(nm,[]).append((N,BQ,A,b,S[b][1],a,S[a][1]))
for nm,v in hits.items():
    print(f"{nm}: {len(v)} strict S4 disagreements")
    for e in v:
        print(f"   N={e[0]} {e[1]}x{e[2]}: count->{e[3]} (runs {e[4]}), S4->{e[5]} (runs {e[6]})"
              .replace("residue-perm-","rp"))
if not hits: print("none")
print()
print("classification check: is twoband-mis a UNION mask?")
print("  D = [0,128) U [1000,1128) -- two disjoint intervals, i.e. a union of two")
print("  bands. It is a union mask. I filed it under 'other' in the previous scan,")
print("  which is a labelling error on my part, not a falsification.")
