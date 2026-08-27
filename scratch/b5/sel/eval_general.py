"""Agreement / regret / contested-cell analysis on the NON-diagonally-invariant
masks -- the ones both 2f and I originally scoped out and d4 falls back to
identity on."""
import sys, time
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
import numpy as np
import general as G
from oracle import tiles_cost, apply_xform

def dense_sinks(g, w, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    return (kv <= q) & ((kv < g) | (q - kv < w))

def dense_docpack(bounds, N):
    q = np.arange(N)[:, None]; kv = np.arange(N)[None, :]
    b = np.array(list(bounds) + [N])
    doc = np.searchsorted(b, np.arange(N), side="right") - 1
    return (kv <= q) & (doc[:, None] == doc[None, :])

def suite(N):
    S = [(f"sinks4+win256", G.Sinks(4, 256), dense_sinks(4, 256, N)),
         (f"sinks16+win128", G.Sinks(16, 128), dense_sinks(16, 128, N)),
         (f"sinks4+win1024", G.Sinks(4, 1024), dense_sinks(4, 1024, N))]
    for bl, nm in [(list(range(0, N, 512)), "docpack-512"),
                   (list(range(0, N, 2048)) if N > 2048 else [0], "docpack-2048"),
                   ([0, 300, 700, 1100, 1500, 1900][:max(2, N // 400)], "docpack-irregular")]:
        S.append((nm, G.DocPack(bl, nm), dense_docpack(bl, N)))
    return S

TILES = [128, 64, 32, 16]
print("=== NON-diagonally-invariant masks: agreement / regret ===")
print(f"{'N':>6}{'cells':>7}{'agree':>8}{'meanReg':>9}{'maxReg':>8}{'declined-costable':>20}")
contested = {0.0: [0, 0], 0.01: [0, 0]}
for N in (1024, 2048):
    cells = agree = dec = 0; regs = []
    for name, spec, M0 in suite(N):
        ocost = {}
        for c in G.CANDIDATES:
            Mt = apply_xform(M0, c)
            if Mt is not None:
                ocost[c] = {(b, a): tiles_cost(Mt, b, a) for b in TILES for a in TILES}
        for BQ in TILES:
            for A in TILES:
                for c in G.CANDIDATES:
                    if G.cost_of(c, spec, N, BQ, A) is None and c in ocost:
                        dec += 1
                pick, _ = G.select(spec, N, BQ, A)
                ranked = sorted((v[(BQ, A)], c) for c, v in ocost.items())
                if pick is None or not ranked:
                    continue
                best = ranked[0][0]
                cells += 1
                agree += (ocost[pick][(BQ, A)] == best)
                r = ocost[pick][(BQ, A)] / best
                regs.append(r)
                gap = (ranked[1][0] / best - 1.0) if len(ranked) > 1 else 0.0
                for m in contested:
                    if gap > m:
                        contested[m][0] += 1
                        contested[m][1] += (ocost[pick][(BQ, A)] == best)
    print(f"{N:>6}{cells:>7}{agree/cells*100:>7.1f}%{sum(regs)/len(regs):>9.4f}"
          f"{max(regs):>8.4f}{dec:>20}")
print("\ncontested-cell agreement (oracle best vs 2nd-best gap > margin):")
for m, (n, a) in sorted(contested.items()):
    print(f"  margin {m*100:>3.0f}%: {a}/{n} = {a/n*100:.1f}%" if n else f"  margin {m*100:>3.0f}%: n/a")

print("\n=== runtime scaling, sinks4+win256, 128x16 ===")
print(f"{'N':>7}{'selector(ms)':>14}{'oracle(ms)':>13}{'speedup':>10}")
for N in (512, 1024, 2048, 4096):
    spec = G.Sinks(4, 256)
    t0 = time.perf_counter(); G.select(spec, N, 128, 16); t1 = time.perf_counter()
    M0 = dense_sinks(4, 256, N)
    for c in G.CANDIDATES:
        Mt = apply_xform(M0, c)
        if Mt is not None:
            tiles_cost(Mt, 128, 16)
    t2 = time.perf_counter()
    print(f"{N:>7}{(t1-t0)*1e3:>14.2f}{(t2-t1)*1e3:>13.2f}{(t2-t1)/(t1-t0):>9.1f}x")
