"""Is the dependency list closed? Derive it from the DEFINITION of the statistic
rather than by enumerating phenomena -- the move that worked for the tile-set
characterisation and failed for both of my phenomenon-enumerations.

  rate = #{cells : argmin_A(cell) != argmin_B(cell)} / #cells

argmin_X(cell) is determined by: objective X, the candidate set, the cell, and
the tie rule. So the statistic is a function of exactly four things:

  K1  criterion pair      (which two objectives)        -- d4 found this
  K2  candidate set       (feasible set)                -- 2f found this
  K3  cell set            (masks x N x tile shapes)     -- b5 found this (pooling)
  K4  tie convention      (strict vs loose)             -- b5 found this (S5)

Closed by the signature of the statistic, not by enumeration. Below: all four
demonstrated moving the SAME underlying question, one knob at a time.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summ(M,BQ,A):
    nq,nk=M.shape
    P=np.pad(M,((0,(-nq)%BQ),(0,(-nk)%A)))
    t=P.reshape(-1,BQ,P.shape[1]//A,A).any(axis=(1,3))
    runs=sum(int((np.diff(np.concatenate(([0],r.astype(int),[0])))==1).sum()) for r in t)
    return {"count":int(t.sum()), "runs":runs, "makespan":int(t.sum(axis=1).max())}

def dsink(g,w,N):
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&((kv<g)|(q-kv<w))
LOCAL = lambda N: dense([Iv(0,256),Ap(0,8,N//8)],N)
SINK  = lambda N: dsink(4,256,N)
ALL_T = [(128,128),(128,64),(128,32),(128,16),(64,64),(64,16),(32,32),(16,16)]
FULL  = ["identity","residue-perm-2","residue-perm-4","residue-perm-8"]

def rate(masks, cands, crit, tiles, strict):
    d=t=0
    for N in (1024,2048):
        for mk in masks:
            M0=mk(N)
            for BQ,A in tiles:
                S={c:summ(m,BQ,A) for c in cands if (m:=apply_xform(M0,c)) is not None}
                if len(S)<2: continue
                t+=1
                a=min(S,key=lambda c:S[c][crit[0]]); b=min(S,key=lambda c:S[c][crit[1]])
                if a!=b and (not strict or S[a][crit[1]]!=S[b][crit[1]]):
                    d+=1
    return f"{d}/{t} = {d/t*100:.0f}%" if t else "n/a"

BASE = dict(masks=[LOCAL,SINK], cands=FULL, crit=("count","runs"),
            tiles=ALL_T, strict=True)
print("BASELINE                                    ", rate(**BASE))
print()
print("K1 criterion pair   count vs MAKESPAN        ",
      rate(**{**BASE, "crit":("count","makespan")}))
print("K2 candidate set    {identity, rp2} only     ",
      rate(**{**BASE, "cands":["identity","residue-perm-2"]}))
print("K3 cell set         local+strided only       ",
      rate(**{**BASE, "masks":[LOCAL]}))
print("K3 cell set         BQ=128 rows only         ",
      rate(**{**BASE, "tiles":[(128,128),(128,64),(128,32),(128,16)]}))
print("K4 tie convention   loose (ties counted)     ",
      rate(**{**BASE, "strict":False}))
print()
print("Same question, five answers. Every knob moves it, and each of the three")
print("of us had independently found exactly the one we happened to vary.")
