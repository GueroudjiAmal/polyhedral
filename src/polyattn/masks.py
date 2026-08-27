"""Mask zoo for the sparse-attention granularity study.

Each mask exposes three things:
  row_cols(q, N)        -> bool[N]  live kv columns for a single query row (brute force / validation)
  union_cols(q0, q1, N) -> bool[N]  union of live kv columns over q in [q0, q1)   (closed form)
  live_count(N)         -> int      exact number of live (q, kv) pairs            (closed form)

union_cols and live_count are written in closed form so the sweep stays O(N^2/BQ)
instead of O(N^2); test_masks.py checks both against brute force.
"""
import numpy as np


class Mask:
    family = "?"

    def row_cols(self, q, N):
        raise NotImplementedError

    def union_cols(self, q0, q1, N):
        raise NotImplementedError

    def live_count(self, N):
        raise NotImplementedError

    # generic fallbacks used only by the validation harness
    def dense(self, N):
        return np.stack([self.row_cols(q, N) for q in range(N)])


def _iv(N, lo, hi):
    """bool[N] for the half-open interval [lo, hi), clipped."""
    out = np.zeros(N, dtype=bool)
    lo = max(0, lo)
    hi = min(N, hi)
    if hi > lo:
        out[lo:hi] = True
    return out


class Causal(Mask):
    family = "causal"
    name = "causal"
    data_dependent = False

    def row_cols(self, q, N):
        return _iv(N, 0, q + 1)

    def union_cols(self, q0, q1, N):
        return _iv(N, 0, q1)

    def live_count(self, N):
        return N * (N + 1) // 2


class SlidingWindow(Mask):
    family = "window"
    data_dependent = False

    def __init__(self, w):
        self.w = w
        self.name = f"window-{w}"

    def row_cols(self, q, N):
        return _iv(N, q - self.w + 1, q + 1)

    def union_cols(self, q0, q1, N):
        return _iv(N, q0 - self.w + 1, q1)

    def live_count(self, N):
        q = np.arange(N, dtype=np.int64)
        return int(np.minimum(q + 1, self.w).sum())


class Dilated(Mask):
    """Causal, attend only to kv with (q - kv) % s == 0."""
    family = "dilated"
    data_dependent = False

    def __init__(self, s):
        self.s = s
        self.name = f"dilated-{s}"

    def row_cols(self, q, N):
        out = np.zeros(N, dtype=bool)
        out[q % self.s: q + 1: self.s] = True
        return out

    def union_cols(self, q0, q1, N):
        s = self.s
        if (q1 - q0) >= s:
            return _iv(N, 0, q1)
        out = _iv(N, q0, q1)
        res = np.unique(np.arange(q0, q1) % s)
        if q0 > 0:
            below = np.arange(0, q0)
            out[:q0] = np.isin(below % s, res)
        return out

    def live_count(self, N):
        q = np.arange(N, dtype=np.int64)
        return int((q // self.s + 1).sum())


class SinksWindow(Mask):
    """StreamingLLM: g attention sinks at the start, plus a causal window of w."""
    family = "sinks+window"
    data_dependent = False

    def __init__(self, g, w):
        self.g, self.w = g, w
        self.name = f"sinks{g}+win{w}"

    def row_cols(self, q, N):
        return _iv(N, 0, min(self.g, q + 1)) | _iv(N, q - self.w + 1, q + 1)

    def union_cols(self, q0, q1, N):
        return _iv(N, 0, min(self.g, q1)) | _iv(N, q0 - self.w + 1, q1)

    def live_count(self, N):
        q = np.arange(N, dtype=np.int64)
        sinks = np.minimum(q + 1, self.g)
        win = np.minimum(q + 1, self.w)
        lo = np.maximum(0, q - self.w + 1)
        overlap = np.maximum(0, np.minimum(self.g, q + 1) - lo)
        return int((sinks + win - overlap).sum())


class PrefixLM(Mask):
    """Bidirectional over the first p tokens, causal after."""
    family = "prefix-lm"
    data_dependent = False

    def __init__(self, p):
        self.p = p
        self.name = f"prefixlm-{p}"

    def row_cols(self, q, N):
        return _iv(N, 0, self.p) if q < self.p else _iv(N, 0, q + 1)

    def union_cols(self, q0, q1, N):
        hi = self.p if q1 <= self.p else max(self.p, q1)
        return _iv(N, 0, hi)

    def live_count(self, N):
        p = min(self.p, N)
        q = np.arange(p, N, dtype=np.int64)
        return int(p * p + (q + 1).sum())


class LocalStrided(Mask):
    """Sparse-Transformer style: causal window w, plus every s-th earlier token."""
    family = "local+strided"
    data_dependent = False

    def __init__(self, w, s):
        self.w, self.s = w, s
        self.name = f"local{w}+str{s}"

    def row_cols(self, q, N):
        out = _iv(N, q - self.w + 1, q + 1)
        out[q % self.s: q + 1: self.s] = True
        return out

    def union_cols(self, q0, q1, N):
        win = _iv(N, q0 - self.w + 1, q1)
        s = self.s
        if (q1 - q0) >= s:
            dil = _iv(N, 0, q1)
        else:
            dil = _iv(N, q0, q1)
            res = np.unique(np.arange(q0, q1) % s)
            if q0 > 0:
                dil[:q0] = np.isin(np.arange(0, q0) % s, res)
        return win | dil

    def live_count(self, N):
        q = np.arange(N, dtype=np.int64)
        win = np.minimum(q + 1, self.w)
        m_min = -(-self.w // self.s)          # ceil(w/s)
        m_max = q // self.s
        stride = np.maximum(0, m_max - m_min + 1)
        return int((win + stride).sum())


class DocPacked(Mask):
    """Packed training batch: causal within each document, no cross-document attention.

    Document boundaries are data-dependent and generally unaligned to any block
    lattice -- this is the tier that needs a runtime inspector.
    """
    family = "doc-packed"
    data_dependent = True

    def __init__(self, mean_len, seed=0):
        self.mean_len, self.seed = mean_len, seed
        self.name = f"docpack-{mean_len}"
        self._cache = {}

    def _bounds(self, N):
        """starts[i] for each position i, and the list of (start, end) docs."""
        if N in self._cache:
            return self._cache[N]
        rng = np.random.default_rng(self.seed)
        lens = []
        tot = 0
        # lognormal doc lengths with the requested mean, floored at 16 tokens
        sigma = 0.6
        mu = np.log(self.mean_len) - sigma ** 2 / 2
        while tot < N:
            L = int(max(16, round(rng.lognormal(mu, sigma))))
            lens.append(min(L, N - tot))
            tot += lens[-1]
        edges = np.concatenate([[0], np.cumsum(lens)])
        docs = [(int(edges[i]), int(edges[i + 1])) for i in range(len(lens))]
        starts = np.zeros(N, dtype=np.int64)
        for s, e in docs:
            starts[s:e] = s
        self._cache[N] = (starts, docs)
        return self._cache[N]

    def row_cols(self, q, N):
        starts, _ = self._bounds(N)
        return _iv(N, int(starts[q]), q + 1)

    def union_cols(self, q0, q1, N):
        _, docs = self._bounds(N)
        out = np.zeros(N, dtype=bool)
        for s, e in docs:
            if e <= q0 or s >= q1:
                continue
            out[s:min(e, q1)] = True
        return out

    def live_count(self, N):
        _, docs = self._bounds(N)
        return int(sum((e - s) * (e - s + 1) // 2 for s, e in docs))


class TwoBand(Mask):
    """Causal, live on two diagonal bands: offsets [0, w1) and [off, off+w2).

    Diagonally invariant, so NOTES sec 5c's symmetry theorem applies. Exists in the
    zoo to break the alignment assumption every other mask satisfies for free:
    with `off` and `w2` multiples of the coarsest tile the max(BQ,A) law holds
    exactly, and with `off` shifted off the grid it fails and the failure GROWS
    with N. See sec 5e -- the zoo being tile-aligned by construction is why six
    rounds of experiments could not see an alignment effect.
    """
    family = "two-band"
    data_dependent = False

    def __init__(self, w1, off, w2=None):
        self.w1, self.off, self.w2 = w1, off, w2 or w1
        self.name = f"twoband-{w1}+{off}"

    def row_cols(self, q, N):
        return (_iv(N, q - self.w1 + 1, q + 1)
                | _iv(N, q - self.off - self.w2 + 1, q - self.off + 1))

    def union_cols(self, q0, q1, N):
        return (_iv(N, q0 - self.w1 + 1, q1)
                | _iv(N, q0 - self.off - self.w2 + 1, q1 - self.off))

    def live_count(self, N):
        q = np.arange(N, dtype=np.int64)
        first = np.minimum(q + 1, self.w1)
        far = q - self.off
        second = np.where(far >= 0, np.minimum(far + 1, self.w2), 0)
        overlap = np.maximum(0, np.minimum(q, far) - np.maximum(q - self.w1 + 1,
                                                                far - self.w2 + 1) + 1)
        overlap = np.where(far >= 0, np.minimum(overlap, np.minimum(first, second)), 0)
        return int((first + second - overlap).sum())


class BidirectionalDocPacked(Mask):
    """Packed batch, attention bidirectional WITHIN each document.

    A real workload -- encoder fine-tuning and embedding-model training -- and
    structurally important here: the mask is SYMMETRIC (M == M.T) but is NOT
    diagonally invariant, so it satisfies the max(BQ, A) law of NOTES sec 5c via
    a different route than the windowed and strided masks do. See sec 5d.
    """
    family = "doc-packed-bidir"
    data_dependent = True

    def __init__(self, mean_len, seed=0, offset=0):
        self._base = DocPacked(mean_len, seed)
        self.mean_len, self.offset = mean_len, offset
        self.name = f"bidoc-{mean_len}" + (f"+{offset}" if offset else "")

    def _docs(self, N):
        _, docs = self._base._bounds(N)
        if not self.offset:
            return docs
        # shift every boundary so no document starts on a tile-grid multiple
        out, o = [], self.offset
        shifted = [(0, min(o, N))] + [(min(s + o, N), min(e + o, N)) for s, e in docs]
        for s, e in shifted:
            if e > s:
                out.append((s, e))
        return out

    def row_cols(self, q, N):
        for s, e in self._docs(N):
            if s <= q < e:
                return _iv(N, s, e)
        return np.zeros(N, dtype=bool)

    def union_cols(self, q0, q1, N):
        out = np.zeros(N, dtype=bool)
        for s, e in self._docs(N):
            if e > q0 and s < q1:
                out[s:e] = True
        return out

    def live_count(self, N):
        return int(sum((e - s) ** 2 for s, e in self._docs(N)))


def zoo():
    return [
        Causal(),
        SlidingWindow(128), SlidingWindow(256), SlidingWindow(512),
        SlidingWindow(1024), SlidingWindow(4096),
        Dilated(2), Dilated(4), Dilated(8),
        SinksWindow(4, 256), SinksWindow(4, 1024),
        LocalStrided(256, 8),
        PrefixLM(1024),
        DocPacked(512), DocPacked(2048),
        BidirectionalDocPacked(512), BidirectionalDocPacked(512, offset=40),
        TwoBand(128, 1024), TwoBand(128, 1000),
    ]
