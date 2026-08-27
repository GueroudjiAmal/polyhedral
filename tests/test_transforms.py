"""Transforms must preserve the mask, and the class A/B cost split must hold."""
import pytest

from polyattn import masks, transforms

N = 512


@pytest.mark.parametrize("name,fn", transforms.candidates())
def test_transform_preserves_every_live_element(mask, name, fn):
    M = mask.dense(N)
    Mt, _ = fn(M)
    if Mt is None:
        pytest.skip(f"{name} not applicable to {mask.name}")
    assert int(Mt.sum()) == int(M.sum())


def test_class_a_tiles_stay_contiguous():
    """A q-independent permutation is applied to K/V once; tiles read A rows."""
    assert transforms.kv_per_tile("A", 0, 1, 16, 16) == 16


def test_class_b_shear_costs_extra_kv_rows():
    """A q-dependent shear makes a BQ x A tile span A + (BQ-1) kv rows."""
    assert transforms.kv_per_tile("B", 1, 1, 16, 16) == 16 + 15


@pytest.mark.parametrize("s", (2, 4, 8))
def test_residue_perm_makes_a_dilated_mask_dense(s):
    """The headline claim of experiment 2, as an assertion."""
    m = masks.Dilated(s)
    M = m.dense(2048)
    live = int(M.sum())
    before = transforms.tile_stats(M, 16, 16)[1] / live
    Mt, _ = transforms.make_residue_perm(s)(M)
    after = transforms.tile_stats(Mt, 16, 16)[1] / live
    assert before > 0.8 * s, "un-permuted dilated waste should be about s"
    assert after < 1.15, "residue permutation should make it near-dense"


def test_single_mask_gallery_renders():
    """fig_mask_gallery with one spec used to hand back a bare Axes."""
    import matplotlib
    matplotlib.use("Agg")
    from polyattn import figures
    fig = figures.fig_mask_gallery(N=128, block=32, specs=[masks.Causal()])
    assert len(fig.axes) == 1


def test_transform_selection_is_grain_dependent():
    """docs/NOTES.md §5b -- the load-bearing result for the compiler framing.

    For a union mask the argmin transform MOVES with the tile shape, so selection
    cannot be a per-mask lookup table: it needs the backend's tile shape as input.
    """
    from polyattn.experiments import grain_dependence as gd
    _, argmin = gd.sweep(masks.LocalStrided(256, 8), N=1024)
    assert len({v for v in argmin.values()}) >= 2, "argmin should move with grain"
    assert argmin[(128, 128)] != argmin[(16, 16)]


def test_pure_lattice_selection_is_grain_independent():
    """The contrast: a pure lattice has one right answer at every tile shape."""
    from polyattn.experiments import grain_dependence as gd
    _, argmin = gd.sweep(masks.Dilated(8), N=1024)
    assert len({v for v in argmin.values()}) == 1
    assert argmin[(16, 16)] == "residue-perm-8"


def test_selector_breaks_ties_toward_the_free_transform():
    """docs/NOTES.md §5h. stridefold-s and residue-perm-s reach IDENTICAL element
    counts on a lattice mask, but stridefold is class B (~136 kv rows per 16x16
    tile vs 16). The cost function cannot see the difference; the tie-break must.
    """
    from polyattn import selector
    for s in (2, 4, 8):
        m = masks.Dilated(s)
        c = selector.costs(m, 2048, 16, 16)
        assert c[f"stridefold-{s}"] == c[f"residue-perm-{s}"], "must be an exact tie"
        assert selector.select(m, 2048, 16, 16) == f"residue-perm-{s}"
