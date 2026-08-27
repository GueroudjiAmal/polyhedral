"""RCM is the published numerical baseline; these pin what it does and doesn't do."""
import numpy as np
import pytest

from polyattn import masks, reorder, transforms

N = 512


def test_rcm_is_a_permutation():
    p = reorder.rcm_order(np.eye(64, dtype=bool) | np.eye(64, k=1, dtype=bool)
                          | np.eye(64, k=-1, dtype=bool))
    assert sorted(p.tolist()) == list(range(64))


@pytest.mark.parametrize("m", [masks.Dilated(4), masks.LocalStrided(64, 8),
                               masks.DocPacked(128), masks.Causal()],
                         ids=lambda m: m.name)
def test_rcm_preserves_every_live_element(m):
    M = m.dense(N)
    Mr, meta = reorder.make_rcm()(M)
    assert int(Mr.sum()) == int(M.sum())
    assert meta[0] == "A", "a permutation of both axes is class A"


@pytest.mark.parametrize("s", (2, 4, 8))
def test_rcm_matches_symbolic_on_a_pure_lattice(s):
    """docs/NOTES.md §5a: RCM finds the residue structure on its own, because the
    symmetrised graph of a pure stride-s mask has exactly s components."""
    m = masks.Dilated(s)
    M = m.dense(1024)
    live = int(M.sum())
    sym, _ = transforms.make_residue_perm(s)(M)
    rcm, _ = reorder.make_rcm()(M)
    w_sym = transforms.tile_stats(sym, 16, 16)[1] / live
    w_rcm = transforms.tile_stats(rcm, 16, 16)[1] / live
    assert w_rcm == pytest.approx(w_sym, rel=0.02)


def test_symbolic_beats_rcm_on_a_union_mask():
    """The surviving claim: when the band connects the graph, RCM cannot
    decompose it, but the predicate still exposes the lattice."""
    m = masks.LocalStrided(256, 8)
    M = m.dense(2048)
    live = int(M.sum())
    sym, _ = transforms.make_residue_perm(8)(M)
    rcm, _ = reorder.make_rcm()(M)
    w_sym = transforms.tile_stats(sym, 16, 16)[1] / live
    w_rcm = transforms.tile_stats(rcm, 16, 16)[1] / live
    assert w_sym < w_rcm / 1.3


def test_rcm_can_regress_on_a_structured_mask():
    """RCM is a heuristic with no guarantee -- on sinks it is worse than doing
    nothing. This is why transform SELECTION is the contribution."""
    m = masks.SinksWindow(4, 256)
    M = m.dense(2048)
    live = int(M.sum())
    base = transforms.tile_stats(M, 128, 32)[1] / live
    rcm, _ = reorder.make_rcm()(M)
    assert transforms.tile_stats(rcm, 128, 32)[1] / live > base
