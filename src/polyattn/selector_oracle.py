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


def is_class_b(name):
    """Class B = q-dependent (shear, stridefold): per-tile gather, un-amortisable.

    Element count cannot distinguish these from class A -- on a lattice mask
    stridefold-s and residue-perm-s are EXACTLY tied -- yet stridefold needs ~136
    kv rows per 16x16 tile against residue-perm's 16. See NOTES sec 5h.
    """
    return name == "shear" or name.startswith("stridefold-")


def oracle(mask, N, BQ, A, candidates=CANDIDATES, prefer_class_a=False):
    """Argmin and costs. `prefer_class_a` breaks ties toward the free transform.

    The default (candidate order) is the convention all three sessions agreed;
    it is also indifferent between a free permutation and one needing 8x the
    memory traffic, and therefore PENALISES a selector that prefers the free one.
    Both conventions are reported by `evaluate` rather than one being chosen.
    """
    c = oracle_costs(mask, N, BQ, A, candidates)
    key = ((lambda k: (c[k], is_class_b(k), CANDIDATES.index(k))) if prefer_class_a
           else (lambda k: (c[k], CANDIDATES.index(k))))
    return min(c, key=key), c


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

    # Regime probes. Each exists to VIOLATE a property that every other mask in
    # the set happens to satisfy -- see PROBES below for which. Do not add a mask
    # here without naming the property it breaks, or the set grows long without
    # growing in coverage, which is exactly how the two holes below survived.
    out += [masks.SinksWindow(4, 8), masks.SinksWindow(4, 4),
            masks.SinksWindow(16, 4), masks.SinksWindow(8, 2),
            masks.DocPacked(8, min_len=2), masks.DocPacked(4, min_len=2),
            masks.DocPacked(0, bounds=[0, 3, 5, 900]),
            masks.BidirectionalDocPacked(8)]

    # random diagonally-invariant offset sets
    rng = np.random.default_rng(seed)
    for i, k in enumerate((50, 200, 600)):
        D = np.sort(rng.choice(4000, size=k, replace=False))
        out.append(_custom(lambda q, kv, D=D: (kv <= q) & np.isin(q - kv, D),
                           f"randD-{k}"))
    return out


#: What each probe mask exists to violate. Every other mask in the set satisfies
#: the property; the probe does not. Both holes recorded here were real bugs in
#: real implementations, found only after someone enumerated what their inputs
#: held constant -- so the annotation is load-bearing, not documentation.
PROBES = {
    "sinks4+win8":   "window NARROWER than the fold depth (w < s)",
    "sinks4+win4":   "w < s, extreme",
    "sinks16+win4":  "w < s with g > w",
    "sinks8+win2":   "w < s, minimal window",
    "docpack-8m2":   "documents SHORTER than the fold depth",
    "docpack-4m2":   "documents shorter than the fold depth, extreme",
    "docpack-b0-3-5-900": "mixed 2-token and 895-token documents",
    "bidoc-8":       "short documents, bidirectional (symmetric, non-invariant)",
    "twoband-128+1000": "diagonal run offset NOT a multiple of the coarsest tile",
    "randD-600":     "unstructured offset set, many runs per row",
    "c2-splitter":   "argmin splits within a max(BQ,A) class",
}


def _max_stride():
    return max(int(c.rsplit("-", 1)[1]) for c in CANDIDATES
               if c.startswith("residue-perm-"))


def _docs(m, N):
    base = getattr(m, "_base", m)
    return [e - s for s, e in base._bounds(N)[1]]


def _n_runs(m, N):
    import numpy as np
    live = m.row_cols(N - 1, N)
    d = np.flatnonzero(live[::-1])
    return 1 if d.size == 0 else 1 + int((np.diff(d) > 1).sum())


#: For each probe, a predicate that checks the cell ACTUALLY EXHIBITS the named
#: property -- not merely that the probe ran. A probe that runs plenty of cells,
#: none of which exhibit its property, is the failure mode a run-count cannot
#: see: two probes in a reviewing session's suite were labelled "non-power-of-two
#: N" and "N not divisible by the fold depth" and neither produced a single cell
#: with either property, because the N values chosen were all divisible by every
#: tile. An unverified label is worse than no label, for the same reason a silent
#: probe is worse than no probe: it is what makes the set look comprehensive.
#: SHAPE checks restate the label and cannot fail for a mask of the right kind;
#: BEHAVIOUR checks can. Where a probe's property has an observable consequence,
#: the verifier tests the consequence. `_shape` marks the ones that remain
#: structural because no cheap behavioural signature exists -- they are weaker
#: and labelled as such rather than left to look equivalent.
PROBE_CHECKS = {
    # behaviour: an s > w candidate must actually cost, and differ from identity
    "sinks4+win8":  lambda m, N: _wide_fold_is_live(m, N),
    "sinks4+win4":  lambda m, N: _wide_fold_is_live(m, N),
    "sinks16+win4": lambda m, N: _wide_fold_is_live(m, N) and m.g > m.w,   # +shape
    "sinks8+win2":  lambda m, N: _wide_fold_is_live(m, N),
    # behaviour: short documents must CHANGE the answer vs the long-document twin
    "docpack-8m2":  lambda m, N: _identity_loses_badly_unlike_long_docs(m, N),
    "docpack-4m2":  lambda m, N: _identity_loses_badly_unlike_long_docs(m, N),
    # behaviour: inherits its ARGMIN from the tiny twin and its REGRET SCALE
    # from the long one -- neither uniform case reaches that combination
    "docpack-b0-3-5-900": lambda m, N: _unlike_both_uniform_twins(m, N),
    # behaviour: symmetric yet NOT obeying the max-law -- the whole point of bidoc
    "bidoc-8":      lambda m, N: _symmetric_but_breaks_max_law(m, N),
    # behaviour: the misaligned twin must break the max-law where aligned holds
    "twoband-128+1000": lambda m, N: _breaks_max_law_where_twin_holds(m, N),
    # behaviour: no fold beats identity. If an "unstructured" set admitted a
    # real fold it would not be unstructured in the sense the label claims.
    "randD-600":    lambda m, N: _no_fold_beats_identity(m, N),
    "c2-splitter":  lambda m, N: _argmin_splits_within_a_max_class(m, N),
}


def _spread(m, N, dims=(128, 64, 32, 16)):
    """Max spread of waste within any max(BQ,A) class. 0 => the max-law holds."""
    live = m.live_count(N)
    by = {}
    for bq in dims:
        for a in dims:
            if N % bq or N % a:
                continue
            w = oracle_costs(m, N, bq, a, ["identity"])["identity"] / live
            by.setdefault(max(bq, a), []).append(w)
    return max((max(v) - min(v)) / max(v) for v in by.values() if v)


def _cost_symmetry(m, N, dims=(128, 32, 16)):
    live = m.live_count(N)
    c = {(bq, a): oracle_costs(m, N, bq, a, ["identity"])["identity"] / live
         for bq in dims for a in dims if not (N % bq or N % a)}
    return max(abs(c[(a, b)] - c[(b, a)]) for a in dims for b in dims
               if (a, b) in c and (b, a) in c)


def _wide_fold_is_live(m, N):
    """A fold deeper than the window must cost, and must not equal identity."""
    from . import masks as _m
    wide = [s for s in (2, 3, 4, 6, 8, 12, 16, 32) if s > m.w]
    if not wide:
        return False
    c = oracle_costs(m, N, 128, 32,
                     ["identity"] + [f"residue-perm-{s}" for s in wide])
    return len(c) > 1 and any(v != c["identity"] for k, v in c.items()
                              if k != "identity")


def _identity_loses_badly_unlike_long_docs(m, N):
    """Short documents must CHANGE something measurable, not merely exist.

    The argmin is the wrong signature -- short and long document packing both
    pick `shear`. What short documents change is how badly the identity fallback
    loses, which is exactly what NOTES sec 5g measured: 6.9x at N=1024 against
    1.02x for the long-document twin.
    """
    from . import masks as _m
    _, c = oracle(m, N, 128, 32)
    _, cl = oracle(_m.DocPacked(512), N, 128, 32)
    short_regret = c["identity"] / min(c.values())
    long_regret = cl["identity"] / min(cl.values())
    return short_regret > 3.0 and short_regret > 3 * long_regret


def _symmetric_but_breaks_max_law(m, N):
    """bidoc's reason to exist: cost symmetry exact, max-law violated."""
    return _cost_symmetry(m, N) < 1e-12 and _spread(m, N) > 0.01


def _no_fold_beats_identity(m, N):
    """An unstructured displacement set must admit no useful residue fold.

    Behavioural, not structural: a control with real lattice structure
    (`local256+str8`) reaches 1.33-2.29x on the same measure, so this check CAN
    fail, which is what distinguishes it from restating "unstructured".
    """
    rp = [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)]
    ok = True
    for BQ, A in ((128, 128), (16, 16)):
        c = oracle_costs(m, N, BQ, A, ["identity"] + rp)
        best = min(v for k, v in c.items() if k != "identity")
        ok &= (c["identity"] / best) < 1.01
    return ok


def _unlike_both_uniform_twins(m, N):
    """"Mixed" must name a BEHAVIOUR, not a shape.

    The mixed mask takes its argmin from the tiny-document twin (shear) and its
    identity-regret scale from the long one (~1.03 against tiny's 64.0). Neither
    uniform case reaches that combination, so the probe exercises something they
    do not.
    """
    from . import masks as _m
    tiny = _m.DocPacked(0, bounds=list(range(0, N, 2)))
    long_ = _m.DocPacked(0, bounds=[0])
    for BQ, A in ((128, 128), (32, 32)):
        bm, cm = oracle(m, N, BQ, A)
        bt, ct = oracle(tiny, N, BQ, A)
        bl, cl = oracle(long_, N, BQ, A)
        rm = cm["identity"] / cm[bm]
        rt = ct["identity"] / ct[bt]
        if bm != bl and rt / rm > 5.0:        # argmin unlike long, scale unlike tiny
            return True
    return False


def _breaks_max_law_where_twin_holds(m, N):
    """The misaligned twoband must violate the max-law where the aligned one obeys it."""
    from . import masks as _m
    return _spread(m, N) > 0.01 and _spread(_m.TwoBand(128, 1024), N) < 1e-12


def _argmin_splits_within_a_max_class(m, N):
    """The c2-splitter's whole reason to exist: two cells sharing max(BQ,A)
    whose oracle argmin differs. Verified, not asserted."""
    cells = [(128, 128), (128, 64), (128, 32), (128, 16), (64, 128), (16, 128)]
    picks = {oracle(m, N, bq, a)[0] for bq, a in cells if not (N % bq or N % a)}
    return len(picks) > 1


def verify_probes(ns=(1024, 1536, 2048), verbose=True):
    """Does every probe actually exhibit the property its label claims?

    Checked over the GRID, not a single N, and a probe passes if the property
    appears at ANY N the harness scores. That is not laxity -- some properties
    are genuinely N-dependent (NOTES sec 5i: the argmin moves with N), so
    `c2-splitter` splits within a max class at N=1024 and 1536 and not at 2048.
    A single-N verifier would have called that probe broken; the grid-wide one
    reports where it fires, which is the honest claim.
    """
    if isinstance(ns, int):
        ns = (ns,)
    by_name = {m.name: m for m in test_masks()}
    out = {}
    for name, prop in PROBES.items():
        m = by_name.get(name)
        chk = PROBE_CHECKS.get(name)
        if m is None:
            out[name] = (False, "mask not in the test set")
        elif chk is None:
            out[name] = (False, "no verifier written for this label")
        else:
            hits = []
            for N in ns:
                try:
                    if chk(m, N):
                        hits.append(N)
                except Exception:
                    pass
            where = ("all N" if len(hits) == len(ns)
                     else f"only N={','.join(map(str, hits))}" if hits else "")
            out[name] = (bool(hits), f"{prop}   [{where}]" if where else prop)
    if verbose:
        for name, (ok, why) in out.items():
            print(f"  {'OK  ' if ok else 'FAIL'} {name:<22} {why}")
        bad = [n for n, (ok, _) in out.items() if not ok]
        print(f"\n  {len(out)-len(bad)}/{len(out)} probes exhibit the property they name"
              + (f"  -- FAILING: {', '.join(bad)}" if bad else ""))
    return out


def uncovered_regimes():
    """Regimes known to exist and NOT probed by this set. Keep it honest.

    Ordered by how likely each is to change a claim rather than add a caveat.
    Add to this list when you decide NOT to probe something -- a long test set
    reads as comprehensive, and only an explicit record of what is missing stops
    that illusion returning at a larger mask count.
    """
    return [
        # 1. Could change a claim, not merely caveat it.
        "DECODE: N_q = 1, or N_q smaller than one query tile. The cost model "
        "asserts BQ | N_q, which fails outright. This is the dominant serving "
        "cost and no experiment here has touched it. NOTE: general non-square "
        "N_q != N_kv is now COVERED -- see tests/test_cost.py, symmetry holds "
        "whenever both tile sizes divide both dimensions.",
        "backward pass -- different access pattern, no selector has been asked",
        "DECODE with the class A permutation maintained per step. The "
        "once-per-forward amortisation that the headline rests on is "
        "PREFILL-ONLY: KV-cache append breaks it, because new keys arrive in "
        "original order and would have to be inserted at permuted positions. No "
        "experiment measures it, and if decode forces a per-token scatter the "
        "measured speedup could go to zero or negative while every experiment in "
        "the set still reads as a success.",
        "multiple transforms composed; the candidate set is single transforms "
        "and the composition search of NOTES sec 5 is a separate mechanism",
        # 2. Caveats.
        "data-dependent masks (learned / top-k selection, KV eviction) -- a hard "
        "boundary, not an implementation gap; see NOTES sec 5f",
        "predicates with no closed-form AP-union, e.g. (q*kv) mod p < t",
        "fold depth exceeding the sequence (s > N), and N below one tile",
        "RAGGED sequence lengths -- a COMPOSITE hole, worse than its parts. "
        "2f's sharpened symmetry hypothesis (padded extents agreeing) is "
        "valuable precisely BECAUSE real N is not a multiple of 128; this "
        "session's kernel ASSERTS divisibility so it can never reach that "
        "regime; this session's selector SILENTLY PREFERRED class B there until "
        "it was made to refuse; and a third session's engine returned numbers "
        "~1% wrong at every ragged N. Three artifacts, one regime, and the only "
        "ones that appeared to handle it were the ones that were wrong. "
        "Separately each looks like a limitation; together it is a hole. "
        "Original note: the symmetry theorem's "
        "sharpest hypothesis -- padded extents agreeing -- is verified against "
        "tile_stats's ZERO-PADDING convention, but gpu/triton_attn.py asserts "
        "divisibility and never sees a ragged tail, so the modelled and "
        "implemented behaviours have never been compared there. Real sequence "
        "lengths are not multiples of 128.",
        "tile shapes that are not powers of two",
        "head- or batch-varying masks",
    ]


def instances(ns=GRID_NS, tiles=GRID_TILES, seed=20260826, stats=None):
    """(mask, N, BQ, A) triples, skipping shapes the cost model cannot express.

    `stats`, if given, is populated with what was skipped and which probe masks
    were actually exercised. **A skipped instance leaves no trace in a score**,
    which is precisely how the ragged-N hole survived in this harness and in a
    reviewing session's outside-the-regime probe simultaneously: both contained
    the same undocumented `continue`, and a reader seeing "882 instances, 85.1%"
    had no way to know how many instances were never created. Documented and
    invisible is how it got here; `evaluate` now prints these counts.
    """
    if stats is not None:
        stats.setdefault("scored", 0)
        stats.setdefault("skipped_ragged", 0)
        stats.setdefault("masks_exercised", set())
    for m in test_masks(seed):
        for N in ns:
            for BQ, A in tiles:
                if N % BQ or N % A:
                    if stats is not None:
                        stats["skipped_ragged"] += 1
                    # DELIBERATE and DOCUMENTED, not incidental. Ragged N is a
                    # known hole across all three artifacts (see
                    # uncovered_regimes) and every cost model refuses it. A skip
                    # like this one, added without thinking, is how another
                    # session's outside-the-regime probe ended up with the
                    # untested regime skipped inside it.
                    continue
                if stats is not None:
                    stats["scored"] += 1
                    stats["masks_exercised"].add(m.name)
                yield m, N, BQ, A


def margin(costs, best):
    """(second-best - best) / best. Zero when the argmin is tied.

    The CONTESTED-CELL filter: score only cells where the oracle's best beats its
    second-best by more than some margin. Two things fall out, the second of which
    was not designed in --

      * an agreement figure is not inflated by cells where any answer is right;
      * the filter is IMMUNE TO THE TIE-BREAK ARTEFACT BY CONSTRUCTION. A class-A
        / class-B tie means best == second-best, so margin == 0, so the cell is
        excluded at any margin > 0. Every disagreement between the two tie
        conventions is exactly such a cell.

    So `prefer_class_a` and this filter solve the same problem by different
    routes, and they answer different questions: the flag says WHICH transform
    was shipped on a tie (which matters, because element count cannot see an 8x
    traffic difference), the filter says the accuracy number is not being carried
    by ties. Keep both.
    """
    vals = sorted(costs.values())
    if len(vals) < 2 or vals[0] == 0:
        return float("inf")
    return (vals[1] - vals[0]) / vals[0]


def evaluate(selector, ns=(1024, 1536, 2048), tiles=None, seed=20260826,
             verbose=True, margins=(0.0, 0.01, 0.05, 0.20)):
    """agreement, mean regret, max regret -- plus the contested-cell breakdown."""
    tiles = tiles or GRID_TILES
    stats = {}
    hits = total = ties = hits_a = 0
    contested = []
    regrets, worst = [], (1.0, None)
    per_family = {}
    for m, N, BQ, A in instances(ns, tiles, seed, stats=stats):
        best, costs = oracle(m, N, BQ, A)
        best_a, _ = oracle(m, N, BQ, A, prefer_class_a=True)
        lo = min(costs.values())
        winners = [k for k, v in costs.items() if v == lo]
        ties += (any(is_class_b(w) for w in winners)
                 and any(not is_class_b(w) for w in winners))
        pick = selector(m, N, BQ, A)
        if pick not in costs:
            pick = "identity"                   # unavailable pick scored as identity
        r = costs[pick] / costs[best]
        total += 1
        hits += (pick == best)
        hits_a += (pick == best_a)
        regrets.append(r)
        if r > worst[0]:
            worst = (r, (m.name, N, BQ, A, pick, best))
        fam = getattr(m, "family", "?")
        d = per_family.setdefault(fam, [0, 0, []])
        d[0] += (pick == best); d[1] += 1; d[2].append(r)
        contested.append((margin(costs, best), pick == best, r))
    res = dict(agreement=hits / total, agreement_class_a=hits_a / total,
               mean_regret=float(np.mean(regrets)), max_regret=worst[0],
               worst_case=worst[1], n=total, class_ab_ties=ties)
    unexercised = sorted(set(PROBES) - stats.get("masks_exercised", set()))
    res["skipped_ragged"] = stats.get("skipped_ragged", 0)
    res["probes_not_exercised"] = unexercised
    if verbose:
        sk = res["skipped_ragged"]
        print(f"  instances scored              {total}")
        if sk:
            print(f"  instances SKIPPED (ragged N)  {sk}   <- never scored by any"
                  " selector, by construction; see uncovered_regimes()")
        else:
            print(f"  instances SKIPPED (ragged N)  0   <- because NO ragged N is in"
                  f" the grid at all: ns={tuple(ns)}")
            print("     The regime is excluded UPSTREAM, not skipped. A zero here is"
                  " not coverage.")
        if unexercised:
            print(f"  PROBES NOT EXERCISED in this run: {', '.join(unexercised)}")
            print("     (a probe that silently does not run is worse than no probe --"
                  " its presence is what makes the set look comprehensive)")
        print(f"  agreement   {res['agreement']*100:.1f}%  ({hits}/{total})"
              f"   [candidate-order ties]")
        print(f"              {res['agreement_class_a']*100:.1f}%  ({hits_a}/{total})"
              f"   [class-A-preferring ties]")
        print(f"  class A/B ties in the cost function: {ties} ({ties/total*100:.1f}%)"
              f" -- element count cannot separate them")
        print(f"  regret      mean {res['mean_regret']:.4f}   max {res['max_regret']:.4f}")
        if worst[1]:
            n_, N_, bq, a, p, b = worst[1]
            print(f"  worst case  {n_} N={N_} {bq}x{a}: picked {p}, best {b}")
        print("  contested cells (margin = how much the best beats second-best):")
        print(f"    {'margin':>8}{'cells':>8}{'agree':>9}{'meanReg':>10}{'maxReg':>9}")
        for mg in margins:
            sub = [c for c in contested if c[0] > mg]
            if not sub:
                continue
            a = sum(x[1] for x in sub) / len(sub)
            rr = [x[2] for x in sub]
            print(f"    {mg*100:>7.0f}%{len(sub):>8}{a*100:>8.1f}%"
                  f"{float(np.mean(rr)):>10.4f}{max(rr):>9.4f}")
        print("  by family:")
        for fam, (h, t, rs) in sorted(per_family.items()):
            print(f"    {fam:<20} agree {h/t*100:5.1f}%  mean regret {np.mean(rs):.4f}")
    return res
