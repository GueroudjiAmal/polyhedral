import sys, time, random
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of
rng = random.Random(20260826)
N = 1024
offs = sorted(rng.sample(range(N), 300))
D = DiagSpec([Iv(o, o+1) for o in offs])
print(f"irregular D, 300 singleton runs, N={N}")
for c in ["identity", "residue-perm-2", "residue-perm-8", "residue-perm-32"]:
    t0 = time.perf_counter(); v = cost_of(c, D, N, 128, 16); t1 = time.perf_counter()
    print(f"  {c:<18}{(t1-t0)*1e3:>10.1f} ms   cost={v}")
