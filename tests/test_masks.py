"""The closed forms are where a silent bug produces a beautiful, wrong result.

Every one is checked against a brute-forced dense mask.
"""
import numpy as np
import pytest

NS = (256, 512)


@pytest.mark.parametrize("N", NS)
def test_live_count_matches_brute_force(mask, N):
    assert mask.live_count(N) == int(mask.dense(N).sum())


@pytest.mark.parametrize("N", NS)
@pytest.mark.parametrize("BQ", (16, 32, 128))
def test_union_cols_matches_brute_force(mask, N, BQ):
    M = mask.dense(N)
    for b in range(N // BQ):
        q0 = b * BQ
        np.testing.assert_array_equal(
            mask.union_cols(q0, q0 + BQ, N), M[q0:q0 + BQ].any(axis=0),
            err_msg=f"{mask.name} N={N} BQ={BQ} block={b}")


def test_docpack_documents_tile_the_sequence():
    from polyattn.masks import DocPacked
    N = 512
    _, docs = DocPacked(96, seed=1)._bounds(N)
    assert docs[0][0] == 0 and docs[-1][1] == N
    assert all(a[1] == b[0] for a, b in zip(docs, docs[1:]))
