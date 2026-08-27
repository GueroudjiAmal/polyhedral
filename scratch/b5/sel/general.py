"""Lifting the diagonal-invariance restriction.

Both 2f and I independently formulated the selector over a displacement set D,
which cannot express sinks (a prefix at ABSOLUTE columns), docpack or bidoc.
That is the zoo's blind spot reproducing in the artefact, and it would narrow the
project's headline claim to translation-invariant masks -- excluding document
packing and StreamingLLM, two of the most deployed patterns.

It does not have to. The requirement is not diagonal invariance; it is only that
the mask supply, in closed form, the union of live kv columns over an ARITHMETIC
PROGRESSION of query rows. Every mask in the zoo can. Given that:

  identity        row-blocks are APs with step 1        -> O(N/BQ * pieces)
  residue-perm-s  a permuted row-block is rq fixed and kq contiguous, i.e. original
                  rows in an AP of step s; each original column interval maps to s
                  intervals in permuted space (kv with kv%s == c lands in block c at
                  position kv//s)                        -> O(N/BQ * s * pieces)
  shear           columns kv-q; union over the BQ rows of a block -> O(N/BQ * BQ)

so the same argmin runs over the whole zoo, not just the lattice/band family.
"""
from blocks import Iv, Ap, count_blocks, billed_cols


class RowSpec:
    """A mask defined by cols(q0, q1, step) -> list of Iv in kv space, the union
    of live columns over rows q0, q0+step, ..., < q1."""
    name = "rowspec"

    def cols(self, q0, q1, step, N):
        raise NotImplementedError


class Sinks(RowSpec):
    """StreamingLLM: g sink columns at absolute 0..g, plus a causal window w."""

    def __init__(self, g, w):
        self.g, self.w = g, w
        self.name = f"sinks{g}+win{w}"

    def cols(self, q0, q1, step, N):
        """Union of live columns over rows q0, q0+step, ..., < q1.

        The window part is contiguous ONLY when step <= w. When the AP step
        exceeds the window width the windows are DISJOINT and the union is one
        interval per row -- so the piece count grows linearly in the number of
        rows. Getting this wrong over-counts by up to 1.76x; my first version
        merged unconditionally and every mask in my validation set had w >= 128
        with s <= 32, so the branch was never exercised.
        """
        qmax = q0 + ((q1 - 1 - q0) // step) * step
        out = [Iv(0, min(self.g, qmax + 1))]
        if step <= self.w:
            out.append(Iv(max(0, q0 - self.w + 1), qmax + 1))
        else:
            q = q0
            while q <= qmax:
                out.append(Iv(max(0, q - self.w + 1), q + 1))
                q += step
        return out


class DocPack(RowSpec):
    """Packed documents: causal within a document, nothing across."""

    def __init__(self, bounds, name="docpack"):
        self.b = list(bounds)              # ascending document start offsets
        self.name = name

    def _doc(self, q):
        lo = 0
        for x in self.b:
            if x <= q:
                lo = x
            else:
                return lo, x
        return lo, None

    def cols(self, q0, q1, step, N):
        """Union over rows q0, q0+step, ..., < q1.

        Must walk the AP ROWS, not the documents. The first version walked
        documents from q0 to qmax and emitted an interval for each, which is
        wrong whenever a document contains NO row of the AP -- i.e. whenever
        documents are shorter than step. That over-counted by up to 5.33x on
        8-token documents under residue-perm-32, and every docpack case in my
        suite (and in d4's) had documents of 128+ against step <= 32, so the
        regime was unreachable by either test set.
        """
        out = []
        qmax = q0 + ((q1 - 1 - q0) // step) * step
        q, cur_lo, cur_hi = q0, None, None
        while q <= qmax:
            lo, _ = self._doc(q)
            if lo == cur_lo:
                cur_hi = q
            else:
                if cur_lo is not None:
                    out.append(Iv(cur_lo, cur_hi + 1))
                cur_lo, cur_hi = lo, q
            q += step
        if cur_lo is not None:
            out.append(Iv(cur_lo, cur_hi + 1))
        return out


def _perm_intervals(ivs, s, N):
    """Map kv-space intervals through residue-perm-s. kv with kv%s == c lands in
    block c at position kv//s, so each interval becomes at most s intervals."""
    B = N // s
    out = []
    for iv in ivs:
        lo, hi = max(0, iv.lo), min(N, iv.hi)
        if hi <= lo:
            continue
        for c in range(s):
            f = lo + ((c - lo) % s)
            if f >= hi:
                continue
            last = hi - 1 - ((hi - 1 - c) % s)
            out.append(Iv(c * B + f // s, c * B + last // s + 1))
    return out


def cost_identity(m, N, BQ, A):
    tot = 0
    for q0 in range(0, N, BQ):
        tot += BQ * A * count_blocks(m.cols(q0, q0 + BQ, 1, N), A, N)
    return tot


def cost_residue_perm(m, N, BQ, A, s):
    if N % s or N % BQ or N % A:
        return None
    B = N // s
    tot = 0
    for Q0 in range(0, N, BQ):
        ivs = []
        rq0, rq1 = Q0 // B, (Q0 + BQ - 1) // B
        for rq in range(rq0, rq1 + 1):
            k0 = max(Q0, rq * B) - rq * B
            k1 = min(Q0 + BQ, (rq + 1) * B) - 1 - rq * B
            ivs += m.cols(k0 * s + rq, k1 * s + rq + 1, s, N)
        tot += BQ * A * count_blocks(_perm_intervals(ivs, s, N), A, N)
    return tot


def cost_shear(m, N, BQ, A):
    jmin, jmax = None, None
    per_block = []
    for q0 in range(0, N, BQ):
        ivs = []
        for q in range(q0, min(q0 + BQ, N)):
            for iv in m.cols(q, q + 1, 1, N):
                lo, hi = max(0, iv.lo), min(N, iv.hi)
                if hi > lo:
                    ivs.append(Iv(lo - q, hi - q))
        if not ivs:
            per_block.append([]); continue
        jmin = min([jmin] + [i.lo for i in ivs]) if jmin is not None else min(i.lo for i in ivs)
        jmax = max([jmax] + [i.hi for i in ivs]) if jmax is not None else max(i.hi for i in ivs)
        per_block.append(ivs)
    if jmin is None:
        return None
    W = jmax - jmin
    tot = 0
    for ivs in per_block:
        if ivs:
            tot += BQ * billed_cols([Iv(i.lo - jmin, i.hi - jmin) for i in ivs], A, W)
    return tot


CANDIDATES = ["identity", "shear"] + [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)]


def cost_of(name, m, N, BQ, A):
    if name == "identity":
        return cost_identity(m, N, BQ, A)
    if name == "shear":
        return cost_shear(m, N, BQ, A)
    if name.startswith("residue-perm-"):
        return cost_residue_perm(m, N, BQ, A, int(name.split("-")[-1]))
    return None


def select(m, N, BQ, A):
    best = None
    for nm in CANDIDATES:
        c = cost_of(nm, m, N, BQ, A)
        if c is None:
            continue
        key = (c, 0 if nm != "shear" else 1, nm)
        if best is None or key < best[0]:
            best = (key, nm, c)
    return (best[1], best[2]) if best else (None, None)


class BiDoc(RowSpec):
    """Bidirectional packed documents: live iff q and kv share a document.
    Not causal, not diagonally invariant, M = M^T. Same AP-row walk as DocPack:
    a document containing no row of the AP must contribute nothing."""

    def __init__(self, bounds, name="bidoc"):
        self.b = list(bounds)
        self.name = name

    def _doc(self, q):
        lo, hi = 0, None
        for x in self.b:
            if x <= q:
                lo = x
            else:
                hi = x
                break
        return lo, hi

    def cols(self, q0, q1, step, N):
        out, seen = [], set()
        qmax = q0 + ((q1 - 1 - q0) // step) * step
        q = q0
        while q <= qmax:
            lo, hi = self._doc(q)
            if lo not in seen:
                seen.add(lo)
                out.append(Iv(lo, hi if hi is not None else N))
            q += step
        return out
