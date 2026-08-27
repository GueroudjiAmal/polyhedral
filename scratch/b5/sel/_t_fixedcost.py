"""Design the test that kills or saves the fixed-cost MAGNITUDE account.

MY EARLIER SPEC WAS INCOHERENT. I told d4 to find a pair with "equal tiles per
program and unequal totals". At a fixed tile shape nprog = (N/BQ)*H is the same
for both variants, so tiles/program = tiles/nprog, and equal tiles/program forces
equal totals. The test as I described it cannot exist.

CORRECT DERIVATION. Fixed-cost model: time = nprog*F + tiles*V, with nprog
transform-independent. Let rho = T_slow/T_fast (the element ratio counting
predicts) and phi = nprog*F/(T_fast*V) = fixed cost relative to the FAST
variant's tile work. Then

    measured_ratio = (phi + rho) / (phi + 1)

phi -> 0 gives measured = rho (no compression); phi -> inf gives 1 (total
compression). And phi = F / (tiles_per_program_fast * V).

CALIBRATE F FROM THE MISS WE HAVE. dilated-8, 16x16, N=4096: rho = 7.79,
measured 3.34 -> phi = (rho - m)/(m - 1) = 4.45/2.34 = 1.90. The fast variant has
16.5 tiles/program, so F/V = 1.90 * 16.5 = 31.4 tiles-equivalent per program.

THE TEST: find a cell where phi is SMALL, i.e. the fast variant has many tiles
per program, while rho stays large. There the account predicts almost NO
shortfall. If a large shortfall persists, the account is dead.
"""
import sys; sys.path.insert(0, ".")
import numpy as np
from blocks import Iv, Ap
from spec import DiagSpec
from xform import cost_of

H = 8
FV = 31.4          # F/V in tiles-equivalent per program, calibrated above

def tiles(pieces, N, BQ, A, cand):
    c = cost_of(cand, DiagSpec(pieces), N, BQ, A)
    return None if c is None else c // (BQ * A)

MASKS = {"dilated-8": (lambda N: [Ap(0,8,N//8)], "residue-perm-8"),
         "dilated-4": (lambda N: [Ap(0,4,N//4)], "residue-perm-4"),
         "dilated-2": (lambda N: [Ap(0,2,N//2)], "residue-perm-2")}

print(f"{'mask':<11}{'N':>6}{'tile':>9}{'T_slow':>9}{'T_fast':>8}{'rho':>7}"
      f"{'tpp_fast':>10}{'phi':>7}{'pred meas':>11}{'compression':>12}")
rows = []
for nm,(mk,cand) in MASKS.items():
    for N in (4096, 8192, 16384):
        for BQ, A in ((16,16),(32,32),(64,64),(128,128),(128,64)):
            p = mk(N)
            Ts = tiles(p, N, BQ, A, "identity"); Tf = tiles(p, N, BQ, A, cand)
            if not Ts or not Tf: continue
            nprog = (N // BQ) * H
            rho = Ts / Tf
            tpp = Tf * H / nprog        # tiles per program for the fast variant
            phi = FV / tpp
            pred = (phi + rho) / (phi + 1)
            rows.append((pred/rho, nm, N, BQ, A, Ts, Tf, rho, tpp, phi, pred))
rows.sort(reverse=True)
for r in rows[:8]:
    _, nm, N, BQ, A, Ts, Tf, rho, tpp, phi, pred = r
    print(f"{nm:<11}{N:>6}{f'{BQ}x{A}':>9}{Ts:>9}{Tf:>8}{rho:>7.2f}"
          f"{tpp:>10.1f}{phi:>7.2f}{pred:>11.2f}{pred/rho*100:>11.0f}%")
print("\n(compression column = predicted measured ratio as a % of the element ratio;")
print(" 100% means fixed cost predicts NO shortfall at all)")
print("\nFor contrast, the cell where the miss was measured:")
p = [Ap(0,8,4096//8)]
Ts, Tf = tiles(p,4096,16,16,"identity"), tiles(p,4096,16,16,"residue-perm-8")
nprog=(4096//16)*H; rho=Ts/Tf; tpp=Tf*H/nprog; phi=FV/tpp; pred=(phi+rho)/(phi+1)
print(f"  dilated-8 4096 16x16: rho {rho:.2f}  tpp {tpp:.1f}  phi {phi:.2f}"
      f"  predicted measured {pred:.2f}  (actual 3.34)")
