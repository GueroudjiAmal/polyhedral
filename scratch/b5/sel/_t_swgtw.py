"""Does my Sinks.cols handle s > w? My validation set had w in {128,256,1024}
and s <= 32, so s > w NEVER OCCURRED. Testing the branch directly."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
import general as G
from oracle import tiles_cost, apply_xform

def dense_sinks(g, w, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & ((kv < g) | (q - kv < w))

bad = n = 0
for g_, w_ in [(4, 8), (4, 16), (16, 32), (4, 4)]:      # w SMALLER than some s
    for N in (1024, 2048):
        M0 = dense_sinks(g_, w_, N)
        spec = G.Sinks(g_, w_)
        for cand in G.CANDIDATES:
            Mt = apply_xform(M0, cand)
            if Mt is None:
                continue
            for BQ in (128, 32, 16):
                for A in (128, 32, 16):
                    got = G.cost_of(cand, spec, N, BQ, A)
                    if got is None:
                        continue
                    exp = tiles_cost(Mt, BQ, A)
                    n += 1
                    if got != exp:
                        bad += 1
                        if bad <= 6:
                            print(f"  FAIL sinks{g_}+win{w_} {cand} N={N} {BQ}x{A}:"
                                  f" got {got} want {exp}  ({got/exp:.3f}x)")
print(f"s>w branch: {n-bad}/{n} exact" + ("" if not bad else "   <-- BROKEN"))
