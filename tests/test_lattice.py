"""The transform should be DERIVED from the predicate, not guessed from a list."""
import pytest

from polyattn import lattice, masks, selector
from polyattn.selector import CANDIDATES


@pytest.mark.parametrize("s", (2, 3, 4, 5, 8, 16, 32))
def test_fold_is_read_off_the_lattice_not_searched(s):
    """O(#runs), no candidate list. Exact for a rank-1 offset lattice."""
    runs = selector.offsets_of(masks.Dilated(s), 2048)
    assert lattice.derive_fold(runs) == s


@pytest.mark.parametrize("m", [masks.SlidingWindow(128), masks.Causal(),
                               masks.TwoBand(128, 1000)], ids=lambda m: m.name)
def test_a_run_wider_than_a_point_spans_Z(m):
    """No single fold applies; the honest answer is 'decompose', not 'identity'."""
    runs = selector.offsets_of(m, 2048)
    assert lattice.derive_fold(runs) == 1


def test_derivation_exposes_what_enumeration_hides():
    """dilated-3 at N=2048: the derived fold is 3, residue-perm-3 needs
    N % 3 == 0, and the enumerating selector answers `identity` with no signal
    that the correct transform existed and was inexpressible."""
    m, N = masks.Dilated(3), 2048
    runs = selector.offsets_of(m, N)
    res = lattice.derive_transform(runs, N, CANDIDATES)
    assert res.fold == 3 and res.name == "residue-perm-3"
    assert not res.expressible and "not divisible" in res.reason
    assert selector.select(m, N, 16, 16) == "identity"      # silently, today

    # and at an N where it IS expressible, derivation and enumeration agree
    N2 = 2049
    runs2 = selector.offsets_of(m, N2)
    assert lattice.derive_transform(runs2, N2, CANDIDATES).expressible


@pytest.mark.parametrize("s", (2, 4, 8, 16))
def test_derivation_agrees_with_enumeration_where_both_apply(s):
    m, N = masks.Dilated(s), 2048
    res = lattice.derive_transform(selector.offsets_of(m, N), N, CANDIDATES)
    assert res.expressible
    assert selector.select(m, N, 16, 16) == res.name
