"""Experiment 3: composition -- split a mask so each part gets its own basis.

Motivation: `local256+str8` stalls at waste 1.24 under any single transform,
because the residue permutation that makes its strided component dense scatters
its local-window component. The two components want different bases.

WHY THIS IS LEGAL.  Attention over a mask that is a DISJOINT union M = P1 + P2
can be computed as separate attentions over P1 and P2 merged with the standard
online-softmax (log-sum-exp) combine -- the same mechanism flash-decoding and
ring attention already use to split the kv axis. Disjointness is essential: an
element counted in two parts would be double-counted in the softmax denominator.
Every decomposition here is peeled to be disjoint by construction, and the union
is asserted to reconstruct M exactly.

WHAT IT COSTS.  Each extra part adds one partial output + LSE pair per query.
That is NOT folded into the element counts -- mixing traffic into a FLOP proxy
would flatter the method. It is reported separately by merge_overhead_fraction().

SEARCH.  All subsets of the shape library up to size 3, peeled in canonical
order, rejecting any decomposition that does not reconstruct M exactly. Every
part is offered every class-A transform, not just the one its shape suggests --
otherwise the k=1 row is an artificially weak baseline and the reported gain is
inflated. Class B (shear) is excluded: experiment 2 showed it costs more traffic
than it saves.
"""
import itertools

import numpy as np

from .. import masks, shapes, transforms

SEARCH_N = 1024          # search here (fast), verify the winner at VERIFY_N
VERIFY_N = 4096
GRAIN = (16, 16)
PERMS = (2, 4, 8, 16)


def part_cost(part, grain=GRAIN):
    """Cheapest free (class A) basis for this part: identity, or any residue perm."""
    best = (transforms.tile_stats(part, *grain)[1], "identity")
    for s in PERMS:
        Mt, _ = transforms.make_residue_perm(s)(part)
        e = transforms.tile_stats(Mt, *grain)[1]
        if e < best[0]:
            best = (e, f"residue-perm-{s}")
    return best


def evaluate(M, combo, grain=GRAIN, inter=None):
    """Peel `combo` against M in canonical order.

    Returns (total elements computed, per-part detail), or None if the
    decomposition does not reconstruct M exactly. `inter` maps shape name to
    (shape & M), precomputed once per mask so the combination loop never
    rebuilds a dense shape.
    """
    combo = sorted(combo, key=lambda s: (shapes.ORDER[s.kind], -(s.p or 0)))
    N = M.shape[0]
    if inter is None:
        inter = {sh.name: sh.dense(N) & M for sh in combo}
    taken = np.zeros_like(M)
    parts = []
    for sh in combo:
        p = inter[sh.name] & ~taken
        if not p.any():
            return None                                   # redundant shape
        taken |= p
        parts.append((sh, p))
    if not np.array_equal(taken, M):
        return None                                       # incomplete cover
    total, detail = 0, []
    for sh, p in parts:
        e, basis = part_cost(p, grain)
        total += e
        detail.append((sh.name, basis, int(p.sum()), e))
    return total, detail


def search(m, N=SEARCH_N, kmax=3, grain=GRAIN):
    M = m.dense(N)
    live = int(M.sum())
    inter = {sh.name: sh.dense(N) & M for sh in shapes.LIBRARY}
    results = []
    for k in range(1, kmax + 1):
        for combo in itertools.combinations(shapes.LIBRARY, k):
            u = inter[combo[0].name]
            for sh in combo[1:]:
                u = u | inter[sh.name]
            if not np.array_equal(u, M):        # cheap coverage prefilter
                continue
            r = evaluate(M, combo, grain, inter)
            if r is None:
                continue
            total, detail = r
            # On ties prefer the decomposition whose shapes NAME the basis they
            # use -- several shapes can peel to the same residual set, and the
            # self-consistent labelling is the interpretable one.
            mismatch = sum(1 for name, basis, _, _ in detail
                           if basis != "identity"
                           and not name.endswith(basis.split("-")[-1]))
            results.append(dict(k=k, waste=total / live, detail=detail,
                                combo=list(combo), mismatch=mismatch))
    results.sort(key=lambda r: (round(r["waste"], 4), r["k"], r["mismatch"]))
    return results, live


def merge_overhead_fraction(k, N, live, delta_waste, d=128):
    """Back-of-envelope: is the log-sum-exp merge cheap relative to what it buys?

    Each extra part adds one partial output + LSE per query: ~(k-1)*N*(d+1)
    elements moved. Expressed in the same units as the saving -- (q,kv) pairs at
    ~2d FLOPs each -- that is ~(k-1)*N*(d+1)/(2d) pair-equivalents, against a
    saving of delta_waste * live pairs. Traffic versus FLOPs is not a clean
    conversion: treat this as an order-of-magnitude check, not a measurement.
    """
    if k <= 1 or delta_waste <= 0:
        return 0.0
    return ((k - 1) * N * (d + 1) / (2 * d)) / (delta_waste * live)


def default_zoo():
    return [masks.LocalStrided(256, 8), masks.LocalStrided(128, 4),
            masks.SinksWindow(4, 256), masks.Dilated(8),
            masks.SlidingWindow(128), masks.Causal()]


def run(zoo=None, N=SEARCH_N, verify_n=VERIFY_N, verbose=True):
    zoo = zoo or default_zoo()
    out = []
    for m in zoo:
        res, _ = search(m, N)
        if not res:
            if verbose:
                print(f"{m.name}: no decomposition in the library reconstructs it")
            continue
        best = res[0]
        singles = [r for r in res if r["k"] == 1]
        single = min(singles, key=lambda r: (round(r["waste"], 4), r["mismatch"]),
                     default=None)

        Mv = m.dense(verify_n)
        live_v = int(Mv.sum())

        def verify(combo):
            e = evaluate(Mv, combo, GRAIN,
                         {sh.name: sh.dense(verify_n) & Mv for sh in combo})
            return (e[0] / live_v, e[1]) if e else (float("nan"), None)

        v_waste, v_detail = verify(best["combo"])
        s_waste, _ = verify(single["combo"]) if single else (float("nan"), None)

        rec = dict(mask=m.name, parts=best["k"], verified_waste=v_waste,
                   single_best=s_waste, detail=v_detail or best["detail"])
        rec["merge_overhead"] = merge_overhead_fraction(
            best["k"], verify_n, live_v, s_waste - v_waste)
        out.append(rec)
        if verbose:
            print(f"\n{m.name}   (all figures at N={verify_n})"
                  f"   best single basis {rec['single_best']:.3f}"
                  f"  ->  decomposed {rec['verified_waste']:.3f}"
                  f"  in {rec['parts']} part(s)")
            for name, basis, nlive, e in rec["detail"]:
                print(f"      {name:<12} basis={basis:<18} live={nlive:>9,}"
                      f"  waste={e/nlive:.3f}")
    return out


def print_table(res):
    print(f"{'mask':<16}{'best single':>13}{'decomposed':>12}{'parts':>7}"
          f"{'gain':>8}{'merge cost':>12}")
    print("-" * 78)
    for r in res:
        mo = f"{r['merge_overhead']*100:.2f}%" if r["parts"] > 1 else "-"
        print(f"{r['mask']:<16}{r['single_best']:>13.3f}{r['verified_waste']:>12.3f}"
              f"{r['parts']:>7}{r['single_best']/r['verified_waste']:>7.2f}x{mo:>12}")
    print(f"\nbest single / decomposed are waste at {GRAIN[0]}x{GRAIN[1]},"
          f" both verified at N={VERIFY_N}")
    print("merge cost = LSE-combine overhead as a fraction of the work saved"
          " (order-of-magnitude)")


if __name__ == "__main__":
    print(f"search N={SEARCH_N}, verify N={VERIFY_N}, grain {GRAIN[0]}x{GRAIN[1]},"
          f" library of {len(shapes.LIBRARY)} shapes, subsets up to 3\n")
    r = run()
    print("\n" + "=" * 78)
    print_table(r)
