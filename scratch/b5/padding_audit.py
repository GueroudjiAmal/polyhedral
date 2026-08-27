"""2f's item: is any EXISTING repo result computed where tile_stats' zero-padding
makes the two padded extents disagree?  If so its asymmetry is an artefact of the
padding convention, not of the mask, and would read as a finding.

Checks the real thing: every (N, BQ, A) each experiment actually uses, AND the
kv-extent each transform actually produces (shear/stridefold return NON-SQUARE
matrices, which is where a mismatch would most plausibly hide).
"""
import numpy as np
from polyattn import masks, transforms

USES = [  # (module, N, tile shapes) as read from the experiment sources
    ("granularity.sweep",      4096,  [(b, b) for b in (128, 64, 32, 16)]),
    ("granularity.product_grid", 16384, [(b, a) for b in (128,64,32,16) for a in (128,64,32,16)]),
    ("reindex",                4096,  [(16, 16), (32, 32)]),
    ("rcm_vs_symbolic",        2048,  [(128,32),(16,16)]),
    ("grain_dependence",       2048,  [(128,128),(128,32),(64,16),(32,32),(16,16)]),
    ("tile_shape_law",         2048,  [(b, a) for b in (128,64,32,16) for a in (128,64,32,16)]),
    ("compose.search",         1024,  [(16, 16)]),
    ("compose.verify",         4096,  [(16, 16)]),
]

def pad(n, t):
    return -(-n // t) * t

print("=== square-mask results (kv extent == N) ===")
bad = 0
for name, N, grains in USES:
    for BQ, A in grains:
        if pad(N, BQ) != pad(N, A):
            print(f"  MISMATCH {name} N={N} {BQ}x{A}: {pad(N,BQ)} vs {pad(N,A)}")
            bad += 1
print(f"  mismatches: {bad}")

print("\n=== transformed masks: kv extent after each transform (NOT N) ===")
print(f"{'mask':<18}{'transform':<18}{'shape':>14}{'grains checked':>34}")
worst = 0
for m in masks.zoo():
    N = 4096
    try:
        M = m.dense(N)
    except Exception as e:
        print(f"  {m.name}: skipped ({e})"); continue
    for tname, fn in transforms.candidates():
        Mt, meta = fn(M)
        if Mt is None:
            continue
        nq, nk = Mt.shape
        flags = []
        for BQ, A in [(16,16),(32,32),(128,32),(128,128)]:
            if pad(nq, BQ) != pad(nk, A):
                flags.append(f"{BQ}x{A}:{pad(nq,BQ)}!={pad(nk,A)}")
        if flags:
            worst += 1
            print(f"{m.name:<18}{tname:<18}{str(Mt.shape):>14}   {' '.join(flags)}")
print(f"  transform/grain combinations with unequal padded extents: {worst}")
print("\nNOTE: unequal extents only threaten the SYMMETRY theorem, which is a claim"
      "\nabout square diagonally-invariant masks. A non-square sheared matrix is not"
      "\nin its scope. What it would threaten is any waste number read off a padded"
      "\ntail -- checked separately below.")

print("\n=== phantom tiles from padding, as a fraction of tiles counted ===")
for m in masks.zoo():
    N = 4096
    M = m.dense(N)
    for tname, fn in transforms.candidates():
        Mt, meta = fn(M)
        if Mt is None:
            continue
        nq, nk = Mt.shape
        for BQ, A in [(16, 16)]:
            if nk % A or nq % BQ:
                t_pad = transforms.tile_stats(Mt, BQ, A)[0]
                t_cut = transforms.tile_stats(Mt[:nq - nq % BQ, :nk - nk % A], BQ, A)[0]
                if t_pad != t_cut:
                    print(f"  {m.name:<18}{tname:<18}{Mt.shape} {BQ}x{A}: "
                          f"padded {t_pad} vs truncated {t_cut} "
                          f"({(t_pad-t_cut)/t_pad*100:.3f}% phantom)")
print("  (no lines above = no result in the repo is affected)")
