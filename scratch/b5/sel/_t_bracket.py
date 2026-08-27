"""Verify 2f's three thresholds from my own counts, then check the model they
assume: time = tiles*c_T + proxy*c_P."""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from oracle import dense, apply_xform

def summ(M,BQ,A):
    nq,nk=M.shape
    P=np.pad(M,((0,(-nq)%BQ),(0,(-nk)%A)))
    t=P.reshape(-1,BQ,P.shape[1]//A,A).any(axis=(1,3))
    runs=sum(int((np.diff(np.concatenate(([0],r.astype(int),[0])))==1).sum()) for r in t)
    return int(t.sum()), runs, int(t.sum(axis=1).max())

def dsink(g,w,N):
    q=np.arange(N)[:,None]; kv=np.arange(N)[None,:]
    return (kv<=q)&((kv<g)|(q-kv<w))

N=1024
CASES=[("sinks4+win256 128x128 {id,rp8}", dsink(4,256,N), (128,128),
        "identity","residue-perm-8", 1),      # 1 = runs proxy
       ("local256+str8 128x128 {id,rp2}", dense([Iv(0,256),Ap(0,8,N//8)],N), (128,128),
        "identity","residue-perm-2", 1),
       ("local256+str8 128x32 {id,rp2,rp4}", dense([Iv(0,256),Ap(0,8,N//8)],N), (128,32),
        "residue-perm-2","residue-perm-4", 2)]  # 2 = makespan proxy

print(f"{'experiment':<36}{'cand A (tiles,proxy)':>22}{'cand B':>18}{'threshold':>12}")
for lbl, M0, (BQ,A), ca, cb, pi in CASES:
    sa = summ(apply_xform(M0, ca), BQ, A)
    sb = summ(apply_xform(M0, cb), BQ, A)
    dT = sb[0]-sa[0]; dP = sa[pi]-sb[pi]
    thr = dT/dP if dP else float('inf')
    print(f"{lbl:<36}{f'{sa[0]},{sa[pi]}':>22}{f'{sb[0]},{sb[pi]}':>18}{thr:>12.2f}")

print("""
2f's thresholds: 9.50, 0.29, 2.67. Mine above -- compare.

TWO OBJECTIONS TO READING THESE AS ONE BRACKET:

(1) THEY ARE NOT THE SAME COEFFICIENT. Experiments 1 and 2 use the RUNS proxy;
    experiment 3 uses MAKESPAN. c_runs and c_makespan are different constants
    with different units. So this is not three probes on one axis -- it is two
    probes bracketing c_runs/c_T (0.29 and 9.50) and ONE probe of
    c_makespan/c_T (2.67). Ordering the three thresholds on a single line
    implies a comparison that the model does not support.

(2) THE ADDITIVE MODEL IS WRONG FOR MAKESPAN SPECIFICALLY. time = tiles*c_T +
    proxy*c_P treats the proxy as additive work. Makespan is a MAX, not a sum:
    in the wave regime the whole point is that runtime tracks the slowest
    program, so time ~ waves * maxtiles * per-tile-cost, and total tiles barely
    enters. The right model there is time ~ ceil(nprog/SMs) * max_tiles * c,
    which is not of the form tiles*c_T + makespan*c_P at all. So experiment 3's
    threshold is derived from a model that does not describe the regime the
    experiment was chosen to probe.

Runs is fine as additive -- a run boundary is a real extra address computation
per row-block, so runs*c_P is a genuine additive term alongside tiles*c_T.""")
