"""The interactive API is what a reader will actually touch first."""
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pytest

from polyattn import masks
from polyattn.explore import Custom, best_transform, split, try_transform, validate, waste_table

N = 256


def test_custom_reproduces_a_builtin_mask():
    ref = masks.SlidingWindow(64)
    got = Custom(lambda q, kv: (kv <= q) & (q - kv < 64), "w64")
    np.testing.assert_array_equal(got.dense(N), ref.dense(N))
    assert got.live_count(N) == ref.live_count(N)


def test_validate_accepts_a_wellformed_custom_mask():
    assert validate(Custom(lambda q, kv: kv <= q, "causal"), N=N)


def test_validate_rejects_an_inconsistent_mask():
    bad = Custom(lambda q, kv: kv <= q, "liar")
    bad.live_count = lambda n: 0                      # deliberately wrong
    assert not validate(bad, N=N)


def test_try_transform_reports_the_class_b_traffic_penalty():
    r = try_transform("window-128", "shear", N=512, draw=False)
    assert r["cls"] == "B" and r["traffic"] > 1.5


def test_try_transform_is_free_for_class_a():
    m = Custom(lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0), "s8")
    r = try_transform(m, "residue-perm-8", N=512, draw=False)
    assert r["cls"] == "A" and r["traffic"] == 1.0 and r["after"] < 1.3


def test_best_transform_ranks_free_first_on_ties():
    m = Custom(lambda q, kv: (kv <= q) & ((q - kv) % 8 == 0), "s8")
    rows = best_transform(m, N=512)
    top_waste = round(rows[0][0], 4)
    assert rows[0][1] == "A"
    assert any(round(w, 4) == top_waste and cls == "B" for w, cls, _, _ in rows)


def test_split_beats_the_best_single_basis_on_a_union_mask():
    """The claim is relative -- absolute waste is inflated at this small N."""
    res = split(masks.LocalStrided(64, 8), N=512, kmax=2, top=1)
    assert res, "no decomposition accepted"
    best = res[0]
    best_single = min(r["waste"] for r in res if r["k"] == 1)
    assert best["k"] == 2
    assert best["waste"] < best_single


def test_waste_table_is_monotone_in_granularity():
    t = waste_table("window-128", N=512)
    assert t[(128, 128)] >= t[(32, 32)] >= t[(16, 16)]
