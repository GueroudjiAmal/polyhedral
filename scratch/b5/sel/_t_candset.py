"""1. Verify 2f: is S4-vs-count disagreement a property of (mask, CANDIDATE SET)?
   2. Test their one surviving mask-level claim -- docpack immunity -- and see
      whether it is empirical or PROVABLE for the whole residue-perm family."""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summ(M,BQ,A):
    nq,nk=M.shape
    P=np.pad(M,((0,(-nq)%BQ),(0,(-nk)%A)))
    t=P.reshape(-1,BQ,P.shape[1]//A,A).any(axis=(1,3))
    runs=sum(int((np.diff(np.concatenate(([0],r.astype(int),[0])))==1).sum()) for r in t)
    return int(t.sum()),runs

TIL=[(128,128),(128,32),(64,64),(32,32),(16,16)]
def ddoc(b,N):
    bb=np.array(list(b)+[N]); d=np.searchsorted(bb,np.arange(N),side="right")-1
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&(d[:,None]==d[None,:])
def dsink(g,w,N):
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&((kv<g)|(q-kv<w))
MASKS={"local256+str8":lambda N: dense([Iv(0,256),Ap(0,8,N//8)],N),
       "twoband-mis":  lambda N: dense([Iv(0,128),Iv(1000,1128)],N),
       "sinks4+win256":lambda N: dsink(4,256,N),
       "docpack-512":  lambda N: ddoc(list(range(0,N,512)),N)}
SETS={"{id,rp2}":["identity","residue-perm-2"],
      "{id,rp2,rp4}":["identity","residue-perm-2","residue-perm-4"],
      "{id..rp8}":["identity","residue-perm-2","residue-perm-4","residue-perm-8"],
      "{id..rp16}":["identity","residue-perm-2","residue-perm-4","residue-perm-8","residue-perm-16"],
      "{id,rp3,rp5}":["identity","residue-perm-3","residue-perm-5"]}

print("S4-vs-count STRICT disagreement rate, MASK FIXED, candidate set varying")
print(f"{'mask':<16}" + "".join(f"{k:>15}" for k in SETS))
for nm,mk in MASKS.items():
    row=""
    for k,cands in SETS.items():
        d=t=0
        for N in (1024,2048):
            M0=mk(N)
            for BQ,A in TIL:
                S={c:summ(m,BQ,A) for c in cands if (m:=apply_xform(M0,c)) is not None}
                if len(S)<2: continue
                t+=1
                b=min(S,key=lambda c:S[c][0]); a=min(S,key=lambda c:S[c][1])
                d+= (a!=b and S[a][1]!=S[b][1])
        row+=f"{f'{d/t*100:.0f}%' if t else '-':>15}"
    print(f"{nm:<16}{row}")

print("\n=== IS DOCPACK IMMUNITY PROVABLE, not just empirical? ===")
print("Claim: for docpack every residue-perm STRICTLY WORSENS BOTH objectives,")
print("so identity is the argmin under both, for ANY subset of residue-perms.")
print(f"{'N':>6}{'tile':>9}{'cand':>16}{'tiles':>9}{'runs':>7}{'both worse?':>13}")
allworse=True
for N in (1024,2048):
    M0=ddoc(list(range(0,N,512)),N)
    for BQ,A in [(128,128),(16,16)]:
        b_t,b_r=summ(M0,BQ,A)
        for c in ["residue-perm-2","residue-perm-3","residue-perm-4","residue-perm-8","residue-perm-16"]:
            m=apply_xform(M0,c)
            if m is None: continue
            t_,r_=summ(m,BQ,A)
            w=(t_>b_t and r_>b_r)
            allworse &= w
            print(f"{N:>6}{f'{BQ}x{A}':>9}{c.replace('residue-perm-','rp'):>16}"
                  f"{t_:>9}{r_:>7}{('yes' if w else 'NO'):>13}")
print(f"\nevery residue-perm strictly worse on BOTH: {allworse}")
print("=> if so, identity is argmin under both objectives for any candidate subset,")
print("   so docpack immunity is STRUCTURAL for this transform family, not sampled.")
