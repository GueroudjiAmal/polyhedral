"""The cost model is the whole comparison; if it is wrong, every number is."""
import pytest

from polyattn import cost, masks

GRAINS = ((128, 128), (32, 32), (16, 16))


@pytest.mark.parametrize("N", (256, 512))
@pytest.mark.parametrize("BQ,A", GRAINS)
def test_cost_matches_dense_reference(mask, N, BQ, A):
    c, _ = cost.cost(mask, N, BQ, A, exact_only=True)
    assert c == cost.dense_cost(mask, N, BQ, A)


@pytest.mark.parametrize("N", (256, 512))
def test_cost_1x1_is_the_live_count(mask, N):
    """cost(1,1) must be the live element count -- the model's lower bound."""
    c, _ = cost.cost(mask, N, 1, 1, exact_only=True)
    assert c == mask.live_count(N)


@pytest.mark.parametrize("N", (256, 512))
def test_finer_granularity_never_costs_more(mask, N):
    coarse, _ = cost.cost(mask, N, 128, 128, exact_only=True)
    fine, _ = cost.cost(mask, N, 16, 16, exact_only=True)
    assert fine <= coarse


def test_row_block_sampling_error_is_negligible():
    """The sweep samples row-blocks above 512 of them. Bound that error."""
    from polyattn import masks
    for m in (masks.SlidingWindow(128), masks.DocPacked(512), masks.Dilated(8)):
        s, was_sampled = cost.cost(m, 16384, 16, 16)
        e, _ = cost.cost(m, 16384, 16, 16, exact_only=True)
        assert was_sampled
        assert abs(s - e) / e < 0.01, f"{m.name} sampling error too large"


def test_single_interval_masks_are_flat_along_each_axis_alone():
    """docs/NOTES.md §3a: for a mask whose per-row-block union is ONE interval,
    refining either tile axis alone buys nothing -- the win needs both."""
    from polyattn import masks
    for m in (masks.SlidingWindow(128), masks.SlidingWindow(256)):
        N = 4096
        live = m.live_count(N)
        base = cost.cost(m, N, 128, 128, exact_only=True)[0] / live
        kv_only = cost.cost(m, N, 128, 16, exact_only=True)[0] / live
        q_only = cost.cost(m, N, 16, 128, exact_only=True)[0] / live
        both = cost.cost(m, N, 16, 16, exact_only=True)[0] / live
        assert kv_only == pytest.approx(base), "KV axis alone should do nothing"
        assert q_only == pytest.approx(base), "query axis alone should do nothing"
        assert both < base * 0.95, "refining both should help"


def test_multi_piece_masks_do_gain_on_the_kv_axis_alone():
    """The genuine ragged-column content: a union of disjoint affine pieces."""
    from polyattn import masks
    N = 4096
    for m in (masks.SinksWindow(4, 256), masks.DocPacked(512)):
        live = m.live_count(N)
        base = cost.cost(m, N, 128, 128, exact_only=True)[0] / live
        kv_only = cost.cost(m, N, 128, 16, exact_only=True)[0] / live
        assert kv_only < base * 0.95, f"{m.name} should gain on the KV axis alone"


def test_sinks_isolates_the_kv_axis_but_docpack_does_not():
    """docs/NOTES.md §3a: only a mask whose extra piece sits at FIXED columns
    isolates mechanism 2 from the small-query-tile confound.

    sinks = prefix + band. The prefix occupies the same columns for every query,
    so widening the row-block does not grow it and BQ is irrelevant; the waste is
    the separate prefix tile, which only a finer A removes.

    docpack boundaries sit at arbitrary positions, so a shorter row-block simply
    straddles fewer of them -- BQ helps too, and the mask cannot separate the two
    effects.
    """
    from polyattn import masks
    N = 4096

    def w(m, bq, a):
        return cost.cost(m, N, bq, a, exact_only=True)[0] / m.live_count(N)

    for m in (masks.SinksWindow(4, 256), masks.SinksWindow(4, 1024)):
        assert w(m, 128, 16) < w(m, 128, 128) * 0.95, "A axis should help"
        assert w(m, 16, 128) == pytest.approx(w(m, 128, 128)), "BQ axis should not"

    for m in (masks.DocPacked(512), masks.DocPacked(2048)):
        assert w(m, 16, 128) < w(m, 128, 128) * 0.98, "docpack gains on BQ too"


def test_diagonal_invariance_predicts_tile_shape_symmetry():
    """docs/NOTES.md §5c. If a mask depends only on (q - kv), waste is symmetric
    in the two tile dimensions and constant within each max(BQ, A) class -- so
    selection reduces to one scalar. Masks with absolute-position structure
    (a fixed-column prefix, document boundaries) break both."""
    from polyattn.experiments import tile_shape_law as law
    for m, invariant in ((masks.LocalStrided(256, 8), True),
                         (masks.Dilated(8), True),
                         (masks.SlidingWindow(128), True),
                         (masks.SinksWindow(4, 256), False),
                         (masks.DocPacked(512), False)):
        g = law.grid(m, N=1024)
        sym = law.symmetry(g)
        worst = max(law.max_law(g).values())
        if invariant:
            assert sym < 0.01, f"{m.name} should be tile-shape symmetric"
            assert worst < 0.01, f"{m.name} should obey the max(BQ,A) law"
        else:
            assert sym > 0.05, f"{m.name} should NOT be symmetric"


def test_symmetry_does_not_imply_the_max_law():
    """docs/NOTES.md §5d. A symmetric mask (M == M.T) tiles identically under
    (BQ,A) and (A,BQ), but that is strictly weaker than being a function of
    max(BQ,A) -- f(BQ,A) = BQ+A is symmetric and is not a function of max.

    Bidirectional packed documents are the real-workload case: perfectly
    symmetric, yet the one-dimensional selection rule does NOT apply to them.
    """
    from polyattn.experiments import tile_shape_law as law
    import numpy as np
    m = masks.BidirectionalDocPacked(512)
    M = m.dense(1024)
    assert np.array_equal(M, M.T), "bidirectional packing should be symmetric"
    g = law.grid(m, N=1024)
    assert law.symmetry(g) < 0.01, "symmetric mask -> symmetric cost"
    assert max(law.max_law(g).values()) > 0.02, "but NOT a function of max(BQ,A)"


def test_near_dense_masks_pass_the_symmetry_test_vacuously():
    """A mask already near waste 1.0 passes every tiling test for lack of
    anything to separate. Screen these out before counting them as evidence."""
    from polyattn.experiments import tile_shape_law as law
    # the vacuity threshold is N-dependent: causal is 1.12 at N=1024 and 1.06 at
    # N=2048, so screen at the N the claim is made at.
    for m in (masks.PrefixLM(1024), masks.Causal()):
        g = law.grid(m, N=2048)
        assert g[(128, 128)][0] < 1.10, f"{m.name} should be near-dense"
        assert law.symmetry(g) < 0.01


def test_max_law_error_is_worst_at_coarse_tiles():
    """docs/NOTES.md §5d. For symmetric-but-not-diagonally-invariant masks the
    max(BQ,A) law is approximately true at fine grains and badly wrong at coarse
    ones -- so a scalar rule misapplied here errs MOST at BQ=128, the regime real
    kernels run in. Pin the direction, not just the failure.
    """
    from polyattn.experiments import tile_shape_law as law
    for m in (masks.BidirectionalDocPacked(512),
              masks.BidirectionalDocPacked(512, offset=40),
              masks.BidirectionalDocPacked(256, seed=3)):
        spread = law.max_law(law.grid(m, N=2048))
        vals = [spread[k] for k in sorted(spread, reverse=True)]   # max=128 -> 16
        assert vals == sorted(vals, reverse=True), f"{m.name} error should shrink"
        assert vals[0] > 0.1, "and be substantial at the coarse end"
        assert vals[-1] < 0.01, "and vanish at the fine end"


def test_diagonal_invariance_gives_symmetry_always():
    """docs/NOTES.md §5e -- now a THEOREM, not an empirical law.

    F(BQ,A) = D + [-(A-1), BQ-1] satisfies F(A,BQ) = F(BQ,A) - (BQ-A), and BQ-A
    is a multiple of g = gcd(BQ,A), so shifting by it is a bijection of gZ. The
    counts are equal, hence w(BQ,A) == w(A,BQ) with no condition on D beyond
    diagonal invariance. Must hold on the max-law counterexamples too.
    """
    from polyattn.experiments import tile_shape_law as law
    for m in (masks.TwoBand(128, 1024), masks.TwoBand(128, 1000),
              masks.TwoBand(24, 500, 24), masks.LocalStrided(256, 8)):
        assert law.symmetry(law.grid(m, N=2048)) < 1e-9, f"{m.name} symmetry is exact"


def test_diagonal_invariance_does_NOT_give_the_max_law():
    """§5e: the claim §5c made. False. Two masks differing only by a 24-token
    shift: the tile-aligned one obeys the law exactly, the misaligned one does
    not, and the violation GROWS with N so it is not a boundary artefact."""
    from polyattn.experiments import tile_shape_law as law
    prev = 0.0
    for N in (2048, 4096, 8192):
        assert max(law.max_law(law.grid(masks.TwoBand(128, 1024), N=N)).values()) < 1e-9
        spread = law.max_law(law.grid(masks.TwoBand(128, 1000), N=N))[128]
        assert spread > 0.3, "misaligned twin must violate the law"
        assert spread > prev, "and the violation must grow with N"
        prev = spread


def test_argmin_splits_within_a_max_class():
    """§5e: selection is a function of (predicate, BQ, A), NOT (predicate,
    max(BQ,A)). Seven cells all with max=128, two different winning transforms."""
    from polyattn import transforms
    from polyattn.explore import Custom
    N = 1024
    m = Custom(lambda q, kv: (kv <= q) & (((q - kv) < 24) |
               (((q - kv) >= 500) & ((q - kv) < 524)) | (((q - kv) % 2) == 0)), "split")
    M = m.dense(N); live = int(M.sum())
    variants = {"identity": M, **{f"rp{s}": transforms.make_residue_perm(s)(M)[0]
                                  for s in (2, 4, 8, 16)}}
    cells = ((128, 128), (128, 64), (128, 32), (128, 16), (64, 128), (32, 128), (16, 128))
    winners = {min(((transforms.tile_stats(V, bq, a)[1] / live, n)
                    for n, V in variants.items()))[1] for bq, a in cells}
    assert len(winners) > 1, "argmin must split within the max=128 class"


def test_symmetry_is_exact_even_for_unbounded_and_random_offsets():
    """§5e: proved by transpose + point reflection, so it needs no hypothesis on
    D. Compared as exact integers -- a ratio can hide a small asymmetry."""
    import numpy as np
    from polyattn.explore import Custom
    rng = np.random.default_rng(0)
    randD = set(rng.choice(1023, size=200, replace=False).tolist())
    dims, N = (128, 32, 16), 1024
    for m in (masks.Causal(), masks.Dilated(8), masks.LocalStrided(256, 8),
              masks.TwoBand(128, 500),
              Custom(lambda q, kv: (kv <= q) & np.isin(q - kv, list(randD)), "randD")):
        c = {(bq, a): cost.cost(m, N, bq, a, exact_only=True)[0]
             for bq in dims for a in dims}
        worst = max(abs(c[(a, b)] - c[(b, a)]) for a in dims for b in dims)
        assert worst == 0, f"{m.name} symmetry must be exact, got {worst}"


def test_unweighted_closed_form_fails_for_unbounded_offset_sets():
    """§5e: pins the limitation that my original verification missed. The
    unweighted form g*|gZ n F| is a bounded-D approximation; on triangular
    support it is stuck at exactly 2x regardless of N."""
    import numpy as np
    from math import gcd

    def unweighted(D, BQ, A):
        g = gcd(BQ, A)
        hits = {m for d in D
                for m in range(((d - A + 1) // g) * g, d + BQ, g) if m >= d - A + 1}
        return g * len(hits)

    for m, expect_ok in ((masks.SlidingWindow(128), True), (masks.Causal(), False)):
        N = 2048
        M = m.dense(N)
        D = {int(d) for d in
             np.unique(np.subtract.outer(np.arange(N), np.arange(N))[M])}
        ratio = unweighted(D, 128, 128) / (cost.cost(m, N, 128, 128,
                                                     exact_only=True)[0] / N)
        if expect_ok:
            assert ratio < 1.05, "bounded D: approximation should be close"
        else:
            assert ratio > 1.9, "unbounded D: approximation should be ~2x off"


def test_docpack_supports_documents_shorter_than_the_fold_depth():
    """docs/NOTES.md §5g. Every docpack case in this project had documents far
    longer than any candidate stride, which hid a real bug in another session's
    engine and made this session's reported max regret a long-document number."""
    import numpy as np
    N = 1024
    m = masks.DocPacked(8, min_len=2)
    _, docs = m._bounds(N)
    lens = [e - s for s, e in docs]
    assert min(lens) < 8, "the probe must actually contain short documents"
    M = np.stack([m.row_cols(q, N) for q in range(N)])
    assert m.live_count(N) == int(M.sum())          # closed form still exact

    explicit = masks.DocPacked(0, bounds=[0, 3, 5, 900])
    _, d2 = explicit._bounds(N)
    assert [e - s for s, e in d2] == [3, 2, 895, 124]


def test_selector_fallback_is_poor_on_short_documents():
    """§5g, pinned so the corrected number cannot silently regress to the old
    optimistic one: identity is NOT near-optimal when documents are short."""
    from polyattn import selector, selector_oracle as so
    m = masks.DocPacked(4, min_len=2)
    best, costs = so.oracle(m, 1024, 128, 32)
    pick = selector.select(m, 1024, 128, 32)
    assert pick == "identity"                        # the stated fallback
    assert costs[pick] / costs[best] > 3.0, "regret here is ~7x, not ~1.06"


def test_symmetry_extends_to_non_square_domains():
    """docs/NOTES.md §5e/§7d. A reviewing session flagged that the symmetry proof
    assumes a square domain -- the point reflection (q,kv) -> (Nq-1-q, Nkv-1-kv)
    carries d to (Nq-Nkv)-d, not -d, so the involution no longer closes.

    It closes anyway, and the reason is that comparing w(BQ,A) with w(A,BQ) on a
    rectangular domain requires BOTH tile sizes to divide BOTH dimensions -- so
    they divide the difference, and the shift (Nkv-Nq) is necessarily a multiple
    of the tile. The hypothesis that saves it is divisibility, which the cost
    model already asserts. Squareness was never the load-bearing assumption.
    """
    import numpy as np
    from polyattn import transforms
    dims = (128, 64, 32, 16)
    preds = [lambda q, kv: kv <= q,
             lambda q, kv: (kv <= q) & (q - kv < 128),
             lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0)]
    for Nq, Nkv in ((1024, 2048), (2048, 1024), (1024, 1536)):
        q = np.arange(Nq)[:, None]
        kv = np.arange(Nkv)[None, :]
        for pred in preds:
            M = pred(q, kv)
            live = int(M.sum())
            for a in dims:
                for b in dims:
                    if a == b or any(n % t for n in (Nq, Nkv) for t in (a, b)):
                        continue
                    w1 = transforms.tile_stats(M, a, b)[1] / live
                    w2 = transforms.tile_stats(M, b, a)[1] / live
                    assert abs(w1 - w2) < 1e-12, f"{Nq}x{Nkv} {a}/{b}"
