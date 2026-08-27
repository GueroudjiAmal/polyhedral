"""Measured complexity of the AP-union primitive, and a check that nothing in the
path ever allocates an N x N array."""
import sys, time, tracemalloc
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np, general as G
from blocks import Iv
from spec import DiagSpec
from xform import select as dselect
from blocks import Ap

print("=== wall clock vs N, all candidates, tile 128x16 ===")
print(f"{'N':>7}{'sinks(ms)':>11}{'docpack(ms)':>13}{'diag lat+band(ms)':>19}")
prev = {}
for N in (1024, 2048, 4096, 8192, 16384):
    s1 = G.Sinks(4, 256)
    t0 = time.perf_counter(); G.select(s1, N, 128, 16); a = (time.perf_counter()-t0)*1e3
    s2 = G.DocPack(list(range(0, N, 512)), "dp")
    t0 = time.perf_counter(); G.select(s2, N, 128, 16); b = (time.perf_counter()-t0)*1e3
    D = DiagSpec([Iv(0, 256), Ap(0, 8, N // 8)])
    t0 = time.perf_counter(); dselect(D, N, 128, 16); c = (time.perf_counter()-t0)*1e3
    print(f"{N:>7}{a:>11.1f}{b:>13.1f}{c:>19.1f}")
    if prev:
        print(f"        ratio vs prev N: {a/prev[0]:.2f}x  {b/prev[1]:.2f}x  {c/prev[2]:.2f}x"
              f"   (2.00x = linear, 4.00x = quadratic)")
    prev = (a, b, c)

print("\n=== peak allocation during one full selection, N=8192 ===")
print("   (an N x N bool mask would be 67.1 MB)")
for nm, fn in [("sinks4+win256", lambda: G.select(G.Sinks(4, 256), 8192, 128, 16)),
               ("docpack-512", lambda: G.select(G.DocPack(list(range(0, 8192, 512))), 8192, 128, 16)),
               ("diag band+lat8", lambda: dselect(DiagSpec([Iv(0, 256), Ap(0, 8, 1024)]), 8192, 128, 16))]:
    tracemalloc.start()
    fn()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print(f"  {nm:<18}peak {peak/1e6:>7.3f} MB")
