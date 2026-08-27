"""Shared brute-force oracle and shared test set for the selector comparison.

Three sessions are building transform selectors independently. This module is the
only thing they agree on: the candidate set, the instances, and what "right"
means. Slow and obviously correct by design -- it materialises everything.

Nothing here may be used by a selector. A selector that calls into this module
is brute force wearing a selector costume.

SHARED CONVENTIONS -- settle these here, not per-implementation:

  1. BILLING OF A TRAILING PARTIAL COLUMN STRIP: **true width**, not a full A.
     Class B transforms (shear, stridefold) return non-square matrices whose kv
     extent need not be a multiple of A. The last column-block is billed at its
     real width (A - pad). This follows transforms.tile_stats, which the oracle
     calls, so the oracle bills true width. A selector billing full A will
     disagree with the oracle on shear over docpack-512 and twoband-128+1000 --
     a convention mismatch, not a result.
  2. TIES: the oracle returns min() over an insertion-ordered dict, so the
     earliest candidate in CANDIDATES wins a tie. Agreement is therefore
     sensitive to tie-breaking; regret is not, which is why regret is the number
     that matters.
  3. INAPPLICABLE CANDIDATES are omitted from the cost dict entirely (a
     transform returning None). A selector naming one is scored as if it had
     said "identity".
"""
import numpy as np

from . import masks, transforms

#: Fixed so the three implementations are comparable.
CANDIDATES = (["identity", "shear"]
              + [f"stridefold-{s}" for s in (2, 4, 8)]
              + [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)])

GRID_NS = (1024, 1536, 2048, 4096)
GRID_TILES = tuple((bq, a) for bq in (128, 64, 32, 16) for a in (128, 64, 32, 16))


def _transform(name):
    if name == "identity":
        return transforms.t_identity
    if name == "shear":
        return transforms.t_shear
    if name.startswith("stridefold-"):
        return transforms.make_stridefold(int(name.split("-")[1]))
    if name.startswith("residue-perm-"):
        return transforms.make_residue_perm(int(name.split("-")[-1]))
    raise KeyError(name)


def oracle_costs(mask, N, BQ, A, candidates=CANDIDATES):
    """{name: elements computed}. Materialises. This is the definition of truth."""
    M = np.stack([mask.row_cols(q, N) for q in range(N)])
    out = {}
    for name in candidates:
        Mt, _ = _transform(name)(M)
        if Mt is None:
            continue                       # transform not applicable to this mask
        out[name] = transforms.tile_stats(Mt, BQ, A)[1]
    return out


def oracle(mask, N, BQ, A, candidates=CANDIDATES):
    c = oracle_costs(mask, N, BQ, A, candidates)
    return min(c, key=c.get), c


# --------------------------------------------------------------- test set ----
def _custom(fn, name):
    from .explore import Custom
    return Custom(fn, name)


def test_masks(seed=20260826):
    """The shared instances. Seeded, so all three sessions hit identical cases."""
    out = list(masks.zoo())

    # b5's adversarial pair and the C2 splitter
    out += [masks.TwoBand(128, 1024), masks.TwoBand(128, 1000),
            _custom(lambda q, kv: (kv <= q) & (((q - kv) < 24) |
                    (((q - kv) >= 500) & ((q - kv) < 524)) | (((q - kv) % 2) == 0)),
                    "c2-splitter")]

    # misaligned variants of every band width -- offsets not multiples of 128
    for w in (64, 128, 256):
        for off in (300, 500, 1000):
            out.append(masks.TwoBand(w, off))

    # random diagonally-invariant offset sets
    rng = np.random.default_rng(seed)
    for i, k in enumerate((50, 200, 600)):
        D = np.sort(rng.choice(4000, size=k, replace=False))
        out.append(_custom(lambda q, kv, D=D: (kv <= q) & np.isin(q - kv, D),
                           f"randD-{k}"))
    return out


def instances(ns=GRID_NS, tiles=GRID_TILES, seed=20260826):
    """(mask, N, BQ, A) triples, skipping shapes the cost model cannot express."""
    for m in test_masks(seed):
        for N in ns:
            for BQ, A in tiles:
                if N % BQ or N % A:
                    continue
                yield m, N, BQ, A


def evaluate(selector, ns=(1024, 1536, 2048), tiles=None, seed=20260826,
             verbose=True):
    """agreement, mean regret, max regret -- the three numbers being compared."""
    tiles = tiles or GRID_TILES
    hits = total = 0
    regrets, worst = [], (1.0, None)
    per_family = {}
    for m, N, BQ, A in instances(ns, tiles, seed):
        best, costs = oracle(m, N, BQ, A)
        pick = selector(m, N, BQ, A)
        if pick not in costs:
            pick = "identity"                   # unavailable pick scored as identity
        r = costs[pick] / costs[best]
        total += 1
        hits += (pick == best)
        regrets.append(r)
        if r > worst[0]:
            worst = (r, (m.name, N, BQ, A, pick, best))
        fam = getattr(m, "family", "?")
        d = per_family.setdefault(fam, [0, 0, []])
        d[0] += (pick == best); d[1] += 1; d[2].append(r)
    res = dict(agreement=hits / total, mean_regret=float(np.mean(regrets)),
               max_regret=worst[0], worst_case=worst[1], n=total)
    if verbose:
        print(f"  agreement   {res['agreement']*100:.1f}%  ({hits}/{total})")
        print(f"  regret      mean {res['mean_regret']:.4f}   max {res['max_regret']:.4f}")
        if worst[1]:
            n_, N_, bq, a, p, b = worst[1]
            print(f"  worst case  {n_} N={N_} {bq}x{a}: picked {p}, best {b}")
        print("  by family:")
        for fam, (h, t, rs) in sorted(per_family.items()):
            print(f"    {fam:<20} agree {h/t*100:5.1f}%  mean regret {np.mean(rs):.4f}")
    return res
