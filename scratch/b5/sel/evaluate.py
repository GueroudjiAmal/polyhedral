"""Selector evaluation: agreement, regret, declines, runtime scaling.

SCOPE, stated up front: this covers DIAGONALLY-INVARIANT predicates
(live(q,kv) <=> q-kv in D). The shared test set also lists sinks, docpack and
bidoc, which are not diagonally invariant; those are NOT covered and are
reported as out-of-scope rather than silently skipped.
"""
import sys, random, time
sys.path.insert(0, "/home/agueroudji/Work/Polyhedral_sparce/scratch/b5/sel")
from blocks import Iv, Ap
from spec import DiagSpec
from xform import CANDIDATES, cost_of, select
from oracle import oracle_select, oracle_cost

rng = random.Random(20260826)


def rand_D(N, kind):
    if kind == "runs":
        pieces, pos = [], rng.randrange(0, 4) * 128
        for _ in range(rng.randint(1, 3)):
            aligned = rng.random() < 0.5
            lo = pos + (0 if aligned else rng.randint(1, 127))
            w = rng.randint(1, 3) * 128 + (0 if aligned else rng.randint(1, 127))
            pieces.append(Iv(lo, min(lo + w, N)))
            pos = lo + w + rng.randint(1, 4) * 128
        return pieces
    if kind == "scatter":
        return [Iv(o, o + 1) for o in sorted(rng.sample(range(N), min(300, N)))]
    s = rng.choice([2, 3, 4, 6, 8])
    return [Iv(0, rng.choice([64, 128, 200])), Ap(0, s, N // s)]


def build_suite(N):
    S = [("causal", [Iv(0, N)]),
         ("window-128", [Iv(0, 128)]),
         ("window-256", [Iv(0, 256)]),
         ("dilated-2", [Ap(0, 2, N // 2)]),
         ("dilated-4", [Ap(0, 4, N // 4)]),
         ("dilated-8", [Ap(0, 8, N // 8)]),
         ("local256+str8", [Iv(0, 256), Ap(0, 8, N // 8)]),
         ("twoband-aligned", [Iv(0, 128), Iv(1024, 1152)]),
         ("twoband-misaligned", [Iv(0, 128), Iv(1000, 1128)]),
         ("c2-splitter", [Iv(0, 24), Iv(500, 524), Ap(0, 2, N // 2)]),
         ("band-mis-300", [Iv(0, 128), Iv(300, 428)]),
         ("band-mis-77", [Iv(0, 64), Iv(77, 205)])]
    for i in range(3):
        S.append((f"rand-runs-{i}", rand_D(N, "runs")))
    for i in range(2):
        S.append((f"rand-lat-{i}", rand_D(N, "lat")))
    S.append(("rand-scatter", rand_D(N, "scatter")))
    return [(n, [p for p in ps if not p.clip(N).empty()]) for n, ps in S]


TILES = [128, 64, 32, 16]
print("=== agreement / regret vs brute-force oracle ===")
print(f"{'N':>6}{'cells':>7}{'agree':>8}{'meanReg':>9}{'maxReg':>8}{'declined-costable':>20}")
allmax, worst = 1.0, None
percase = {}
for N in (1024, 1536, 2048):
    cells = agree = dec = 0
    regs = []
    for name, pieces in build_suite(N):
        D = DiagSpec(pieces)
        for BQ in TILES:
            for A in TILES:
                if N % BQ or N % A:
                    continue
                for c in CANDIDATES:
                    if cost_of(c, D, N, BQ, A) is None and \
                       oracle_cost(pieces, N, BQ, A, c) is not None:
                        dec += 1
                pick, _ = select(D, N, BQ, A)
                opick, oc = oracle_select(pieces, N, BQ, A, CANDIDATES)
                if pick is None or oc is None:
                    continue
                cells += 1
                agree += (pick == opick)
                r = oracle_cost(pieces, N, BQ, A, pick) / oc if oc else 1.0
                regs.append(r)
                percase.setdefault(name, []).append(r)
                if r > allmax:
                    allmax, worst = r, (name, N, BQ, A, pick, opick, r)
    print(f"{N:>6}{cells:>7}{agree/cells*100:>7.1f}%{sum(regs)/len(regs):>9.4f}"
          f"{max(regs):>8.4f}{dec:>20}")
if worst:
    n, N, BQ, A, p, o, r = worst
    print(f"  worst regret {r:.4f}: {n} N={N} {BQ}x{A} picked {p}, oracle {o}")
else:
    print("  worst regret 1.0000 -- never picked a suboptimal transform")

print("\n  per-mask max regret (only rows != 1.0000 shown):")
clean = 0
for k, v in sorted(percase.items()):
    if max(v) > 1.0 + 1e-12:
        print(f"    {k:<22}{max(v):.4f}")
    else:
        clean += 1
print(f"    ({clean} masks at exactly 1.0000)")

print("\n=== runtime scaling in N (selector vs oracle), local256+str8, 128x16 ===")
print(f"{'N':>7}{'selector(ms)':>14}{'oracle(ms)':>13}{'speedup':>10}")
for N in (512, 1024, 2048, 4096):
    pieces = [Iv(0, 256), Ap(0, 8, N // 8)]
    D = DiagSpec(pieces)
    t0 = time.perf_counter(); select(D, N, 128, 16); t1 = time.perf_counter()
    oracle_select(pieces, N, 128, 16, CANDIDATES); t2 = time.perf_counter()
    print(f"{N:>7}{(t1-t0)*1e3:>14.2f}{(t2-t1)*1e3:>13.2f}{(t2-t1)/(t1-t0):>9.1f}x")
