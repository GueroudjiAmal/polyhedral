"""Reproduce 2f's per-mask S4 table including docpack and sinks, then test their
refined condition AS A PREDICTOR rather than endorsing it.

2f: the condition is not 'union' but 'the mask has BOTH a component a permutation
helps AND a component it scatters'. Operationalised from the predicate:
    HELPS    = D contains a lattice (an AP of stride >= 2), which residue-perm collapses
    SCATTERS = D contains a band (an interval of width >= 2), which it shreds
Predict a disagreement iff BOTH present.
"""
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
TIL=[(128,128),(128,64),(128,32),(128,16),(64,64),(64,16),(32,32),(16,16)]

def dsink(g,w,N):
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&((kv<g)|(q-kv<w))
def ddoc(b,N):
    bb=np.array(list(b)+[N]); d=np.searchsorted(bb,np.arange(N),side="right")-1
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&(d[:,None]==d[None,:])

# (name, builder, has_lattice, has_band)
MASKS=[("local256+str8", lambda N: dense([Iv(0,256),Ap(0,8,N//8)],N), True,  True),
       ("local128+str4", lambda N: dense([Iv(0,128),Ap(0,4,N//4)],N), True,  True),
       ("twoband-mis",   lambda N: dense([Iv(0,128),Iv(1000,1128)],N),False, True),
       ("sinks4+win256", lambda N: dsink(4,256,N),                    False, True),
       ("docpack-512",   lambda N: ddoc(list(range(0,N,512)),N),      False, True),
       ("dilated-8",     lambda N: dense([Ap(0,8,N//8)],N),           True,  False),
       ("window-128",    lambda N: dense([Iv(0,128)],N),              False, True),
       ("lat8+lat3",     lambda N: dense([Ap(0,8,N//8),Ap(0,3,N//3)],N), True, False)]

print(f"{'mask':<16}{'lat':>5}{'band':>6}{'cells':>7}{'strict':>8}{'rate':>7}"
      f"{'predicted':>11}{'':>5}")
rows=[]
for nm,mk,lat,band in MASKS:
    cells=strict=0
    for N in (1024,2048):
        M0=mk(N)
        for BQ,A in TIL:
            S={c:summ(m,BQ,A) for c in CANDS if (m:=apply_xform(M0,c)) is not None}
            if len(S)<2: continue
            cells+=1
            b=min(S,key=lambda c:S[c][0]); a=min(S,key=lambda c:S[c][1])
            strict += (a!=b and S[a][1]!=S[b][1])
    pred = lat and band
    got  = strict>0
    rows.append((nm,pred,got))
    print(f"{nm:<16}{str(lat):>5}{str(band):>6}{cells:>7}{strict:>8}"
          f"{strict/cells*100:>6.0f}%{('DISAGREE' if pred else 'clean'):>11}"
          f"{'  ok' if pred==got else '  PREDICTOR WRONG':>5}")
bad=[r for r in rows if r[1]!=r[2]]
print(f"\npredictor: {len(rows)-len(bad)}/{len(rows)} correct")
for nm,p,g in bad:
    print(f"  WRONG on {nm}: predicted {'disagree' if p else 'clean'}, observed "
          f"{'disagree' if g else 'clean'}")
