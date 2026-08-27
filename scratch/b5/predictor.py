"""Corrected applicability condition for the max-law, tested as a PREDICTOR on
randomly generated diagonally-invariant masks it was not derived from.

CONDITION (checkable from the predicate, no matrix):
  Let D = {d : mask lives at q-kv = d}, G = coarsest tile in the set considered.
  ALIGNED-AND-SEPARATED(D, G):
    every maximal run [l, r] of D has l % G == 0 and (r-l+1) % G == 0,
    and every gap between consecutive runs is >= 2G-1.
  Claim: ALIGNED-AND-SEPARATED  =>  cost/row = sum(w_i) + k*max(BQ,A),
  hence the max-law.  Otherwise the law may fail.
"""
import numpy as np
from polyattn.explore import Custom
from polyattn import cost
TILES = [128, 64, 32, 16]; G = 128
rng = np.random.default_rng(0)

def runs(D):
    D = np.array(sorted(D)); brk = np.where(np.diff(D) > 1)[0]
    st = np.concatenate([[0], brk+1]); en = np.concatenate([brk, [len(D)-1]])
    return [(int(D[a]), int(D[b])) for a, b in zip(st, en)]

def predict(D):
    R = runs(D)
    for l, r in R:
        if l % G or (r-l+1) % G:
            return False
    for (l1, r1), (l2, _) in zip(R, R[1:]):
        if l2 - r1 - 1 < 2*G - 1:
            return False
    return True

def measure(D, N=4096):
    Dl = list(D)
    m = Custom(lambda q, kv: (kv <= q) & np.isin(q-kv, Dl), "d")
    live = m.live_count(N)
    g = {(bq,a): cost.cost(m,N,bq,a,exact_only=True)[0]/live for bq in TILES for a in TILES}
    sym = max(abs(g[(bq,a)]-g[(a,bq)]) for bq in TILES for a in TILES)
    spread = max(max(v)-min(v) for v in
                 ([g[(bq,a)] for bq in TILES for a in TILES if max(bq,a)==mx] for mx in TILES))
    return sym, spread

def rand_D():
    k = rng.integers(1, 4); D = []
    pos = int(rng.integers(0, 4))*G
    for _ in range(k):
        aligned = rng.random() < 0.5
        l = pos + (0 if aligned else int(rng.integers(1, G)))
        w = int(rng.integers(1, 4))*G + (0 if aligned else int(rng.integers(1, G)))
        D += list(range(l, l+w))
        pos = l + w + int(rng.integers(1, 5))*G
    return sorted(set(D))

print(f"{'#':>3}{'runs':<34}{'predict':>9}{'sym':>8}{'spread':>10}{'':>4}")
tp=tn=fp=fn=0; symmax=0.0
for i in range(24):
    D = rand_D(); p = predict(D); sym, sp = measure(D); symmax = max(symmax, sym)
    holds = sp < 1e-9
    ok = (p == holds)
    tp += p and holds; tn += (not p) and (not holds)
    fp += p and not holds; fn += (not p) and holds
    print(f"{i:>3}{str(runs(D))[:33]:<34}{'HOLD' if p else 'FAIL':>9}"
          f"{sym:>8.4f}{sp:>10.4f}   {'ok' if ok else 'PREDICTOR WRONG'}")
print(f"\npredictor: correct-hold {tp}, correct-fail {tn}, "
      f"FALSE-HOLD {fp} (unsafe), false-fail {fn} (conservative)")
print(f"max transpose asymmetry over all 24 masks: {symmax:.6f}  "
      f"({'SYMMETRY THEOREM HOLDS' if symmax < 1e-9 else 'SYMMETRY BROKEN'})")

print("\n=== POSITIVE SIDE: multi-run ALIGNED+SEPARATED masks (predicted HOLD) ===")
pos = [
    [(0,127),(1024,1151)], [(0,255),(512,767),(1536,1791)],
    [(128,255),(896,1151)], [(0,127),(384,511),(1024,1279)],
    [(256,383),(1024,1151),(2048,2303)], [(0,383),(768,895)],
]
bad = 0
for R in pos:
    D = [d for l,r in R for d in range(l,r+1)]
    p = predict(D); sym, sp = measure(D)
    ok = p and sp < 1e-9
    bad += not ok
    print(f"{str(R):<46}{'HOLD' if p else 'FAIL':>6}{sym:>8.4f}{sp:>10.4f}"
          f"   {'ok' if ok else 'PREDICTOR WRONG'}")
print(f"positive-side failures: {bad}/{len(pos)}")
