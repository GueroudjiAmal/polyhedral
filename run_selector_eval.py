"""Evaluate polyattn.selector against the shared oracle. Three numbers + scaling.

Single pass over the instances -- the oracle is O(N^2) per cell, so calling it
twice was the difference between minutes and an hour.
"""
import time

import numpy as np

from polyattn import masks, selector, selector_oracle as so

NS = (1024, 1536, 2048)
TILES = [(128, 128), (128, 32), (128, 16), (64, 64), (64, 16), (32, 32), (16, 16)]

print("=" * 74)
print("SELECTOR: polyattn.selector   (session d4)")
print(f"instances: {len(so.test_masks())} masks x {NS} x {len(TILES)} tile shapes")
print("=" * 74, flush=True)

rows, t_sel = [], 0.0
for m, N, BQ, A in so.instances(NS, TILES):
    best, costs = so.oracle(m, N, BQ, A)
    t0 = time.perf_counter()
    pick = selector.select(m, N, BQ, A)
    t_sel += time.perf_counter() - t0
    if pick not in costs:
        pick = "identity"
    rows.append((m.name, getattr(m, "family", "?"), N, BQ, A, pick, best,
                 costs[pick] / costs[best], selector.offsets_of(m, N) is not None))

agree = np.mean([r[5] == r[6] for r in rows])
reg = np.array([r[7] for r in rows])
w = max(rows, key=lambda r: r[7])
print(f"\n  1. agreement   {agree*100:.1f}%   ({sum(r[5]==r[6] for r in rows)}/{len(rows)})")
print(f"  2. regret      mean {reg.mean():.4f}   max {reg.max():.4f}")
print(f"     worst case  {w[0]} N={w[2]} {w[3]}x{w[4]}: picked {w[5]}, best {w[6]}")

print("\n--- diagonally invariant vs not (the stated limitation) ---", flush=True)
for label, sel in (("invariant", True), ("NOT invariant", False)):
    sub = [r for r in rows if r[8] is sel]
    if not sub:
        continue
    a = np.mean([r[5] == r[6] for r in sub]); rr = np.array([r[7] for r in sub])
    print(f"  {label:<16} agree {a*100:5.1f}% ({len(sub):4d} cells)"
          f"   mean regret {rr.mean():.4f}   max {rr.max():.4f}")

print("\n--- by mask family ---")
fams = {}
for r in rows:
    fams.setdefault(r[1], []).append(r)
for f, sub in sorted(fams.items()):
    a = np.mean([r[5] == r[6] for r in sub]); rr = np.array([r[7] for r in sub])
    print(f"  {f:<20} agree {a*100:5.1f}%   mean regret {rr.mean():.4f}   max {rr.max():.4f}")

print("\n--- aligned vs misaligned two-band (2f's on-record prediction) ---")
for label, pick in (("aligned (1024)", lambda n: "1024" in n),
                    ("misaligned", lambda n: any(s in n for s in ("300", "500", "1000")))):
    sub = [r for r in rows if r[1] == "two-band" and pick(r[0])]
    if sub:
        a = np.mean([r[5] == r[6] for r in sub]); rr = np.array([r[7] for r in sub])
        print(f"  {label:<18} agree {a*100:5.1f}% ({len(sub)} cells)"
              f"   mean regret {rr.mean():.4f}   max {rr.max():.4f}")

print(f"\n  3. selector runtime: {t_sel/len(rows)*1e3:.2f} ms/instance average", flush=True)
print("\n--- runtime scaling in N (local256+str8, 128x32) ---")
print(f"{'N':>8}{'selector ms':>14}{'oracle ms':>12}{'ratio':>10}")
mm = masks.LocalStrided(256, 8)
for N in (1024, 2048, 4096, 8192, 16384):
    t0 = time.perf_counter()
    for _ in range(3):
        selector.select(mm, N, 128, 32)
    ts = (time.perf_counter() - t0) / 3 * 1e3
    t0 = time.perf_counter(); so.oracle(mm, N, 128, 32); to = (time.perf_counter() - t0) * 1e3
    print(f"{N:>8}{ts:>14.2f}{to:>12.1f}{to/ts:>9.0f}x", flush=True)
