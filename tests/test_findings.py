"""Executable assertions for the numbers docs/NOTES.md actually CLAIMS.

Every other suite in this repo validates CORRECTNESS -- engine against oracle,
closed form against brute force, 564 of them. None of them defended a FINDING.
The two look identical from a passing test count and are entirely different
things: a correctness suite can be comprehensive while every recorded result in
the log is reproducible only by re-running a script and reading the output, which
means any of them could silently change and nothing would fail.

That gap is what let a probe drift from the finding it was created for
(NOTES §7b): probes assert PROPERTIES, results are FINDINGS, and a property can
keep holding long after it has stopped defending anything. The fix is not another
check -- it is to make the finding itself the assertion, so the link is identity
rather than documentation. Delete the mask and the finding's test stops running,
loudly, at collection.

IF ONE OF THESE FAILS, THE CODE IS PROBABLY NOT BROKEN. A recorded claim has
changed. Update NOTES and the assertion together, or explain why the old number
was wrong -- do not quietly edit the constant.
"""
import numpy as np
import pytest

from polyattn import cost, masks, reorder, selector, selector_oracle as so, transforms


def _waste(m, N, BQ, A):
    return cost.cost(m, N, BQ, A)[0] / m.live_count(N)


# ---------------------------------------------------------------- §3a / §3a-bis
def test_finding_mechanism_2_is_dead_against_a_realistic_baseline():
    """§3a-bis KILLED mechanism 2. Against Binary Block Masking's actual 128x32
    tile the remaining headroom is 1.01-1.04x -- noise. Against FlexAttention's
    128x128 default it looked like 1.15-1.28x, which was picking the weakest
    available opponent."""
    N = 16384
    for m, flex_gain, bbm_gain in ((masks.SinksWindow(4, 256), 1.28, 1.04),
                                   (masks.DocPacked(512), 1.15, 1.02),
                                   (masks.SinksWindow(4, 1024), 1.09, 1.01)):
        w128, w32, w16 = (_waste(m, N, 128, a) for a in (128, 32, 16))
        assert w128 / w16 == pytest.approx(flex_gain, abs=0.02), f"{m.name} vs FlexAttn"
        assert w32 / w16 == pytest.approx(bbm_gain, abs=0.02), f"{m.name} vs BBM"


def test_finding_single_interval_masks_need_both_tile_axes():
    """§3a. window-128 sits at 2.00 waste at 128x128, 128x16 AND 16x128, and
    reaches 1.12 only at 16x16. Neither axis alone buys anything."""
    m, N = masks.SlidingWindow(128), 16384
    assert _waste(m, N, 128, 128) == pytest.approx(2.00, abs=0.01)
    assert _waste(m, N, 128, 16) == pytest.approx(2.00, abs=0.01)
    assert _waste(m, N, 16, 128) == pytest.approx(2.00, abs=0.01)
    assert _waste(m, N, 16, 16) == pytest.approx(1.12, abs=0.01)


# ------------------------------------------------------------------------- §4
def test_finding_class_b_shear_is_a_trap():
    """§4. shear drives window waste to ~1.00 but raises kv rows per 16x16 tile
    from 16 to 31 -- ~1.94x traffic for ~1.12x fewer elements. Net loss."""
    assert transforms.kv_per_tile("A", 0, 1, 16, 16) == 16
    assert transforms.kv_per_tile("B", 1, 1, 16, 16) == 31
    M = masks.SlidingWindow(128).dense(4096)
    live = int(M.sum())
    before = transforms.tile_stats(M, 16, 16)[1] / live
    after = transforms.tile_stats(transforms.t_shear(M)[0], 16, 16)[1] / live
    assert before / after == pytest.approx(1.12, abs=0.02)


def test_finding_stridefold_and_residue_perm_tie_on_elements_only():
    """§5h. EXACTLY tied on element count, 136 vs 16 kv rows per tile. The cost
    function cannot separate them; the tie-break must."""
    c = selector.costs(masks.Dilated(8), 4096, 16, 16)
    assert c["stridefold-8"] == c["residue-perm-8"]
    assert transforms.kv_per_tile("B", 1, 8, 16, 16) == 136
    assert selector.select(masks.Dilated(8), 4096, 16, 16) == "residue-perm-8"


# ------------------------------------------------------------------- §5a / RCM
def test_finding_rcm_ties_symbolic_on_lattices_and_loses_on_unions():
    """§5a. RCM reconstructs the residue permutation on a pure lattice (the
    symmetrised graph has s components), so mechanism 1's headline is NOT novel
    against it. On a union mask the band connects the graph, RCM cannot
    decompose, and symbolic wins -- that is the surviving claim."""
    N = 2048
    for s in (4, 8):
        M = masks.Dilated(s).dense(N)
        live = int(M.sum())
        sym = transforms.tile_stats(transforms.make_residue_perm(s)(M)[0], 16, 16)[1]
        rcm = transforms.tile_stats(reorder.make_rcm()(M)[0], 16, 16)[1]
        assert rcm / live == pytest.approx(sym / live, rel=0.02), "must TIE"

    M = masks.LocalStrided(256, 8).dense(N)
    live = int(M.sum())
    sym = transforms.tile_stats(transforms.make_residue_perm(8)(M)[0], 16, 16)[1] / live
    rcm = transforms.tile_stats(reorder.make_rcm()(M)[0], 16, 16)[1] / live
    assert rcm / sym > 1.3, "symbolic must beat RCM on a union mask"


def test_finding_rcm_regresses_on_a_structured_mask():
    """§5a. RCM is a heuristic with no guarantee -- worse than doing nothing on
    sinks. This is why SELECTION is the contribution, not any one transform."""
    M = masks.SinksWindow(4, 256).dense(2048)
    base = transforms.tile_stats(M, 128, 32)[1]
    assert transforms.tile_stats(reorder.make_rcm()(M)[0], 128, 32)[1] > base


# --------------------------------------------------------------------- §5e/§5i
def test_finding_the_max_law_is_false_and_the_violation_grows_with_N():
    """§5e KILLED the max-law. Two masks differing by a 24-token shift: the
    aligned one obeys it exactly, the misaligned one does not, and the violation
    GROWS with N so it is not a boundary artefact."""
    from polyattn.experiments import tile_shape_law as law
    prev = 0.0
    for N, expect in ((2048, 0.3287), (4096, 0.3918), (8192, 0.4163)):
        assert max(law.max_law(law.grid(masks.TwoBand(128, 1024), N=N)).values()) < 1e-9
        got = law.max_law(law.grid(masks.TwoBand(128, 1000), N=N))[128]
        assert got == pytest.approx(expect, abs=0.01)
        assert got > prev
        prev = got


def test_finding_the_argmin_depends_on_N():
    """§5i. Selection is a function of (predicate, BQ, A, N) -- so a lookup table
    needs a third input, and N changes per request."""
    m = masks.LocalStrided(256, 8)
    for N, tile, want in ((1024, (128, 128), "residue-perm-2"),
                          (2048, (128, 128), "residue-perm-2"),
                          (4096, (128, 128), "residue-perm-4"),
                          (1024, (16, 16), "residue-perm-4"),
                          (2048, (16, 16), "residue-perm-8"),
                          (4096, (16, 16), "residue-perm-8")):
        cands = ["identity"] + [f"residue-perm-{s}" for s in (2, 4, 8)]
        assert so.oracle(m, N, *tile, candidates=cands)[0] == want, f"N={N} {tile}"


# ------------------------------------------------------------------------ §5g
def test_finding_the_identity_fallback_loses_badly_on_short_documents():
    """§5g. This session's reported max regret of 1.0562 was a LONG-DOCUMENT
    number. On short documents the declared fallback loses ~6.9x."""
    _, c = so.oracle(masks.DocPacked(8, min_len=2), 1024, 128, 32)
    assert c["identity"] / min(c.values()) == pytest.approx(6.909, abs=0.05)
    _, cl = so.oracle(masks.DocPacked(512), 1024, 128, 32)
    assert cl["identity"] / min(cl.values()) == pytest.approx(1.024, abs=0.02)


# ------------------------------------------------------------------------ §5d
def test_finding_symmetry_does_not_imply_the_max_law():
    """§5d. bidoc is exactly symmetric and violates the max-law -- so M = M^T
    buys cost symmetry only, and the scalar rule does not apply to it."""
    from polyattn.experiments import tile_shape_law as law
    g = law.grid(masks.BidirectionalDocPacked(512), N=2048)
    assert law.symmetry(g) < 1e-12
    assert law.max_law(g)[128] == pytest.approx(0.223, abs=0.01)


def test_finding_class_b_cost_is_a_bracket_not_an_estimate():
    """§4/§5. kv_per_tile counts DISTINCT rows, which assumes intra-tile sharing
    the kernel cannot express: for kv' = (kv - a*q)/s the row index is a [BQ, A]
    MATRIX, so K is not shared across a tile's query rows and a tl.dot kernel
    loads BQ*A rows' worth. Lower and upper bounds are 8-64x apart.

    Recorded so the optimistic end (1.94x at 16x16) is never quoted bare again.
    """
    for BQ, A, gap in ((128, 128, 64.3), (128, 32, 25.8), (16, 16, 8.3)):
        lo = transforms.kv_per_tile("B", 1, 1, BQ, A)     # distinct rows
        hi = BQ * A                                        # no sharing at all
        assert hi / lo == pytest.approx(gap, rel=0.02)
        assert lo > transforms.kv_per_tile("A", 0, 1, BQ, A)


def test_finding_the_class_ab_criterion_is_the_row_index_shape():
    """§4. Sharper than amortisability and derivable from the predicate: a == 0
    gives a vector row index (K shared across the tile), a != 0 gives a [BQ, A]
    matrix (it cannot be). Selector-visible, not an implementation detail."""
    import numpy as np
    M = np.zeros((64, 64), dtype=bool)
    M[0, 0] = True
    for name, fn in transforms.candidates():
        out, meta = fn(M)
        if meta is None:
            continue
        kind, a, _ = meta
        assert (a != 0) == (kind == "B"), f"{name}: index shape must track the class"


def test_finding_disagreement_rate_depends_on_the_candidate_set():
    """§7e. Three sessions each proposed a condition for WHICH MASKS show a
    counting-vs-hardware disagreement, and all three failed on different masks.
    The rate is not a property of the mask: hold the mask and the criteria fixed,
    vary only the candidate list, and it moves 0% -> 38% -> 0%.

    This is why exp3 and exp7 must quote their candidate set with any number they
    report -- a small count from a narrow set is not a small effect.
    """
    import numpy as np

    def stats(M, BQ, A):
        t = M.reshape(M.shape[0] // BQ, BQ, M.shape[1] // A, A).any(axis=(1, 3))
        return int(t.sum()), int(t.sum(axis=1).max())

    m, tiles = masks.LocalStrided(256, 8), [(bq, a) for bq in (128, 64, 32, 16)
                                            for a in (128, 64, 32, 16)]
    rates = {}
    for label, folds in (("narrow", (2,)), ("wide", (2, 4, 8)), ("coprime", (3, 5))):
        dis = tot = 0
        for N in (1024, 2048):
            M0 = np.stack([m.row_cols(i, N) for i in range(N)])
            cand = {"id": M0}
            for s in folds:
                cand[f"rp{s}"] = transforms.make_residue_perm(s)(M0)[0]
            for BQ, A in tiles:
                st = {k: stats(v, BQ, A) for k, v in cand.items()}
                tot += 1
                dis += (min(st, key=lambda k: st[k][0])
                        != min(st, key=lambda k: st[k][1]))
        rates[label] = dis / tot
    assert rates["narrow"] == 0.0, "a single fold has nothing to disagree with"
    assert rates["wide"] > 0.3, "adding rp4/rp8 creates disagreements"
    assert rates["coprime"] == 0.0, "rp3/rp5 do not collapse a stride-8 lattice"
