"""b5 pre-registered predictions for the Polaris runs.

Exact element counts from the symbolic engine -- no materialisation, no
measurement. Written BEFORE any timing exists so a wall-clock number tests the
model rather than being explained by it.
"""
import sys
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of, select

def D(pieces): return DiagSpec(pieces)
def dil(s, N): return D([Ap(0, s, N // s)])
def win(w, N): return D([Iv(0, w)])
def ls(N): return D([Iv(0, 256), Ap(0, 8, N // 8)])

print("b5 PREDICTIONS -- exact element counts, generated before any GPU run\n")

print("== exp2 / GO-NO-GO: dilated-8, identity vs residue-perm-8 ==")
print(f"{'N':>7}{'tile':>9}{'identity':>15}{'rp8':>15}{'ratio':>8}")
for N in (4096, 8192, 16384):
    for BQ, A in ((128, 128), (128, 64), (64, 64), (16, 16)):
        i = cost_of("identity", dil(8, N), N, BQ, A)
        r = cost_of("residue-perm-8", dil(8, N), N, BQ, A)
        if i and r:
            print(f"{N:>7}{f'{BQ}x{A}':>9}{i:>15,}{r:>15,}{i/r:>8.2f}x")

print("\n== exp1 / tile shape: window-128, does wall clock track elements? ==")
print(f"{'N':>7}{'tile':>9}{'elements':>15}{'vs 128x128':>12}")
for N in (4096, 16384):
    base = cost_of("identity", win(128, N), N, 128, 128)
    for BQ, A in ((128, 128), (128, 16), (16, 128), (16, 16)):
        c = cost_of("identity", win(128, N), N, BQ, A)
        print(f"{N:>7}{f'{BQ}x{A}':>9}{c:>15,}{base/c:>11.2f}x")

print("\n== exp3 / selection: local256+str8 argmin per tile shape, N=4096 ==")
print(f"{'BQ\\\\A':>6}" + "".join(f"{a:>22}" for a in (128, 64, 32, 16)))
for BQ in (128, 64, 32, 16):
    row = ""
    for A in (128, 64, 32, 16):
        p, c = select(ls(4096), 4096, BQ, A)
        row += f"{p + ' ' + format(c, ',')  :>22}"
    print(f"{BQ:>6}{row}")

print("""
== PRE-REGISTERED EXPECTATIONS -- what I expect to be WRONG, and why ==

1. exp2 dilated-8 + residue-perm-8 is the one I expect to HOLD in wall clock,
   within say 25% of the element ratio. Class A keeps every tile contiguous and
   the permutation is applied once per layer, so there is no omitted traffic
   term. If this MISSES badly, the class A / class B distinction that the whole
   taxonomy rests on is wrong, and that outranks every other result.

2. exp1 window-128 at 16x16 will UNDERPERFORM its predicted ratio, probably by a
   lot. My cost function prices occupancy, MMA efficiency and per-tile softmax
   statistics at exactly zero, and all three get worse as BQ falls. I expect the
   measured gain to be well under the element-count gain, and I would not be
   surprised by a wall-clock LOSS at BQ=16. NOTES 3a already says this bet has
   bad odds; I am recording that I agree before seeing the number.

3. exp3 selection: my argmin will disagree with the wall-clock argmin on masks
   where the tie is close, and I expect my model to lose wherever the winner is
   class B, for the same missing-traffic reason. Where the winner is class A I
   expect agreement.

4. SPECIFIC FALSIFIER I would accept as killing my engine: any configuration
   where the measured element count executed by the kernel differs from my
   predicted element count. That is a claim about counting, not about time, and
   it has no excuse available. Timing disagreements indict the cost MODEL;
   count disagreements indict the ENGINE.

5. TIE HAZARD, measured: 41.1% of my costed instances have a class-A/class-B
   tie on elements. My selector breaks toward class A by construction. Under the
   shared oracle's candidate-order tie-break my agreement reads 73.2% instead of
   100%, and the element cost is IDENTICAL in all 90 disagreements. So a low
   agreement score for me on the shared metric is expected and is not an error.
""")
