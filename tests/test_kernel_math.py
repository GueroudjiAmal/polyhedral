"""The Triton kernel's online-softmax recurrence, simulated in numpy.

The kernel itself needs a GPU. Its ARITHMETIC does not, and the one bug found in
it by review was a pure arithmetic bug -- so it is pinned here, in the suite that
runs everywhere, rather than left to a scarce ALCF queue slot to discover.

Bug: m_i initialised to -inf. A tile is visited when ANY row of the block reaches
it, so a row can be entirely dead in the FIRST tile it sees; then
max(qk) = -inf, m_new = -inf, and alpha = exp(-inf - -inf) = exp(nan). Every
later accumulation is poisoned, and the `l_i > 0` guard catches l_i == 0 but not
l_i == nan, so the row returns NaN instead of zero.
"""
import numpy as np
import pytest

from polyattn import masks

N, D = 256, 16


def _tiles(M, BQ, A):
    """Same tile selection blockindex.build makes: any live element in the tile."""
    t = M.reshape(N // BQ, BQ, N // A, A).any(axis=(1, 3))
    return [np.flatnonzero(t[i]) for i in range(N // BQ)]


def _kernel_math(M, q, k, v, BQ, A, m_init):
    """Replicates _attn_fwd exactly, including the l_i > 0 guard at the end."""
    out = np.zeros((N, D), dtype=np.float64)
    scale = D ** -0.5
    for i, kvs in enumerate(_tiles(M, BQ, A)):
        rows = slice(i * BQ, (i + 1) * BQ)
        m_i = np.full(BQ, m_init)
        l_i = np.zeros(BQ)
        acc = np.zeros((BQ, D))
        for j in kvs:
            cols = slice(j * A, (j + 1) * A)
            qk = (q[rows] @ k[cols].T) * scale
            qk = np.where(M[rows, cols], qk, -np.inf)
            m_new = np.maximum(m_i, qk.max(axis=1))
            with np.errstate(invalid="ignore"):
                alpha = np.exp(m_i - m_new)
                p = np.exp(qk - m_new[:, None])
            l_i = l_i * alpha + p.sum(axis=1)
            acc = acc * alpha[:, None] + p @ v[cols]
            m_i = m_new
        out[rows] = acc / np.where(l_i > 0, l_i, 1.0)[:, None]
    return out


def _reference(M, q, k, v):
    s = (q @ k.T) * D ** -0.5
    s = np.where(M, s, -np.inf)
    e = np.exp(s - s.max(axis=1, keepdims=True))
    e = np.where(M, e, 0.0)
    den = e.sum(axis=1, keepdims=True)
    return np.where(den > 0, e / np.where(den > 0, den, 1.0), 0.0) @ v


CASES = [(masks.SlidingWindow(64), 128, 128), (masks.SlidingWindow(64), 128, 16),
         (masks.SlidingWindow(32), 64, 16), (masks.Dilated(8), 128, 16),
         (masks.Causal(), 128, 32)]


@pytest.mark.parametrize("m,BQ,A", CASES, ids=lambda x: str(x))
def test_finite_sentinel_matches_dense_reference(m, BQ, A):
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((N, D)) for _ in range(3))
    M = np.stack([m.row_cols(i, N) for i in range(N)])
    got = _kernel_math(M, q, k, v, BQ, A, -1e30)
    assert np.isfinite(got).all(), "finite sentinel must never produce NaN"
    assert np.abs(got - _reference(M, q, k, v)).max() < 1e-9


def test_minus_inf_init_is_the_bug_and_is_reachable():
    """Guards the fix: if someone reverts to -inf this must start NaN-ing again."""
    rng = np.random.default_rng(0)
    q, k, v = (rng.standard_normal((N, D)) for _ in range(3))
    M = np.stack([masks.SlidingWindow(64).row_cols(i, N) for i in range(N)])
    assert not np.isfinite(_kernel_math(M, q, k, v, 128, 128, -np.inf)).all()


def test_a_fully_dead_query_row_returns_zero():
    """A row live in no tile at all must give zeros, not NaN -- what l_i > 0 is for."""
    rng = np.random.default_rng(1)
    q, k, v = (rng.standard_normal((N, D)) for _ in range(3))
    M = np.stack([masks.SlidingWindow(64).row_cols(i, N) for i in range(N)])
    M[5] = False
    got = _kernel_math(M, q, k, v, 128, 128, -1e30)
    assert np.isfinite(got).all() and np.abs(got[5]).max() == 0.0
