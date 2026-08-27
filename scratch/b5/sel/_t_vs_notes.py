"""Cross-check my engine against NOTES 5c's published table for local256+str8
at N=2048 (their argmin/waste per tile shape), and then vary N -- because their
table is at ONE N and 'argmin depends on N too' would be a further finding."""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import select, cost_of

def ls(N): return DiagSpec([Iv(0, 256), Ap(0, 8, N // 8)])
def live(N): return ls(N).live(N)

# NOTES 5c, N=2048, class-A candidates only (rp2/rp4/rp8), argmin (waste)
PUB = {(128,128):("rp2",2.41),(128,64):("rp2",2.41),(128,32):("rp2",2.41),(128,16):("rp2",2.41),
       (64,64):("rp4",1.91),(64,32):("rp4",1.91),(64,16):("rp4",1.91),
       (32,32):("rp4",1.60),(32,16):("rp4",1.60),(16,16):("rp8",1.33)}
CANDS = ["identity","residue-perm-2","residue-perm-4","residue-perm-8"]
print("cross-check vs NOTES 5c table (N=2048, class-A candidates only)")
print(f"{'tile':>9}{'published':>16}{'mine':>22}{'match':>8}")
ok = True
for (BQ,A),(pn,pw) in sorted(PUB.items()):
    best = min((cost_of(c, ls(2048), 2048, BQ, A), c) for c in CANDS
               if cost_of(c, ls(2048), 2048, BQ, A) is not None)
    w = best[0]/live(2048)
    short = best[1].replace("residue-perm-","rp")
    m = (short == pn and abs(w-pw) < 0.006)
    ok &= m
    print(f"{f'{BQ}x{A}':>9}{f'{pn} {pw:.2f}':>16}{f'{short} {w:.3f}':>22}{'yes' if m else 'NO':>8}")
print(f"\n{'reproduces NOTES 5c exactly' if ok else 'DISAGREES with NOTES 5c'}")

print("\nargmin vs N (class-A candidates, waste in parens) -- is the table N-dependent?")
print(f"{'tile':>9}" + "".join(f"{f'N={N}':>18}" for N in (1024,2048,4096,8192)))
for BQ,A in [(128,128),(64,64),(32,32),(16,16)]:
    row=""
    for N in (1024,2048,4096,8192):
        best = min((cost_of(c, ls(N), N, BQ, A), c) for c in CANDS
                   if cost_of(c, ls(N), N, BQ, A) is not None)
        row += f"{best[1].replace('residue-perm-','rp') + f' ({best[0]/live(N):.2f})':>18}"
    print(f"{f'{BQ}x{A}':>9}{row}")
