"""Cost of a TRANSFORMED diagonal mask, computed symbolically.

The load-bearing observation, and the reason a closed form survives a class-A
permutation at all:

  RESIDUE-PERM-s MAPS A DIAGONALLY-INVARIANT PROBLEM TO s^2 DIAGONALLY-INVARIANT
  SUBPROBLEMS.  Sorting both axes by (i mod s, i div s) sends index i to block
  r = i mod s, position k = i div s.  With q = kq*s + rq and kv = kkv*s + rkv,

      q - kv = s*(kq - kkv) + (rq - rkv)

  so block (rq, rkv) is a B x B mask (B = N/s) over (kq, kkv) whose offset set is
      D_delta = {(d - delta)/s : d in D, d = delta (mod s)},  delta = rq - rkv,
  which depends on delta alone.  There are (s - |delta|) blocks per delta, so

      cost = sum over delta in [-(s-1), s-1] of (s-|delta|) * cost_B(D_delta)

  Each term is the analytic diagonal engine on a B x B domain.  Total O(s * N/s)
  = O(N), no matrix anywhere.  This is what makes the permuted branch cheap, and
  it is why residue-perm-s collapses a stride-s lattice: only delta = 0 mod s
  survives, and there D_delta is a contiguous run.

SHEAR / STRIDEFOLD are class B and become column-uniform: every query row has the
same transformed column set, so the cost is a per-row-block block-count over a
reflected (or divided) copy of D.  O(N/BQ * pieces).
"""
from math import gcd
from blocks import Iv, Ap, count_blocks, billed_cols
from spec import DiagSpec

CANDIDATES = (["identity", "shear"]
              + [f"stridefold-{s}" for s in (2, 4, 8)]
              + [f"residue-perm-{s}" for s in (2, 3, 4, 6, 8, 12, 16, 32)])


def _shift_div(pieces, delta, s):
    """{(d - delta)/s : d in pieces, d = delta (mod s)} as Iv/Ap pieces."""
    out = []
    for p in pieces:
        if isinstance(p, Iv):
            if p.empty():
                continue
            first = p.lo + ((delta - p.lo) % s)
            if first < p.hi:
                cnt = (p.hi - 1 - first) // s + 1
                out.append(Iv((first - delta) // s, (first - delta) // s + cnt))
        else:
            if p.count <= 0:
                continue
            step, st = p.stride, p.start
            d0 = gcd(step, s)
            if (delta - st) % d0:
                continue
            per = s // d0
            for m in range(min(per, p.count)):
                if (st + m * step - delta) % s == 0:
                    cnt = (p.count - 1 - m) // per + 1
                    out.append(Ap((st + m * step - delta) // s,
                                  per * step // s, cnt))
    return out


def cost_residue_perm(D, N, BQ, A, s):
    """Exact, via the s^2 block decomposition. Requires N % s == 0 and tiles to
    fit inside a residue block (B % BQ == 0 and B % A == 0)."""
    if N % s or s < 2:
        return None
    B = N // s
    if B % BQ or B % A:
        return _rp_straddle(D, N, BQ, A, s)
    total = 0
    for delta in range(-(s - 1), s):
        sub = _shift_div(D.pieces, delta, s)
        if not sub:
            continue
        c = DiagSpec(sub).cost(B, BQ, A)
        if c:
            total += (s - abs(delta)) * c
    return total


def _rp_straddle(D, N, BQ, A, s):
    """Tiles larger than a residue block. A permuted index x = c*B + k, so a tile
    covers a known set of residue blocks; a tile is live iff SOME (row-residue,
    col-residue) pair has D_delta meeting the corresponding k-difference window.
    Only reached when BQ > B or A > B, i.e. s > N/BQ, so the row-block count is
    below s and this stays O(s * N / A)."""
    B = N // s
    if N % BQ or N % A:
        return None
    subs = {}
    for delta in range(-(s - 1), s):
        sub = _shift_div(D.pieces, delta, s)
        subs[delta] = DiagSpec(sub) if sub else None
    total = 0
    for Q0 in range(0, N, BQ):
        rq = range(Q0 // B, (Q0 + BQ - 1) // B + 1)
        for K0 in range(0, N, A):
            ck = range(K0 // B, (K0 + A - 1) // B + 1)
            live = False
            for cq in rq:
                k_lo, k_hi = max(Q0, cq * B) - cq * B, min(Q0 + BQ, (cq + 1) * B) - 1 - cq * B
                for cv in ck:
                    d = subs.get(cq - cv)
                    if d is None:
                        continue
                    l_lo = max(K0, cv * B) - cv * B
                    l_hi = min(K0 + A, (cv + 1) * B) - 1 - cv * B
                    if d.hits(k_lo - l_hi, k_hi - l_lo):
                        live = True
                        break
                if live:
                    break
            if live:
                total += BQ * A
    return total


def _reflect(pieces, jmin, dmax_allowed):
    """j = -d - jmin for d in D with d <= dmax_allowed."""
    out = []
    for p in pieces:
        if isinstance(p, Iv):
            lo, hi = p.lo, min(p.hi, dmax_allowed + 1)
            if hi > lo:
                out.append(Iv(-(hi - 1) - jmin, -lo - jmin + 1))
        else:
            q = Ap(p.start, p.stride, p.count)
            if q.count <= 0:
                continue
            n = min(q.count, (dmax_allowed - q.start) // q.stride + 1)
            if n > 0:
                last = q.start + (n - 1) * q.stride
                out.append(Ap(-last - jmin, q.stride, n))
    return out


def cost_shear(D, N, BQ, A):
    lo, _ = D.span()
    hi = D.max_le(N - 1)                 # offsets beyond N-1 are never realised
    if hi is None or hi < lo or lo < 0:
        return None                      # shear here assumes a causal offset set
    jmin, W = -hi, hi - lo + 1
    tot = 0
    for q0 in range(0, N, BQ):
        pieces = _reflect(D.pieces, jmin, q0 + BQ - 1)
        if pieces:
            tot += BQ * billed_cols(pieces, A, W)
    return tot


def cost_stridefold(D, N, BQ, A, s):
    lo, _ = D.span()
    hi = D.max_le(N - 1)
    if hi is None or hi < lo or lo < 0:
        return None
    for p in D.pieces:                   # every offset must sit on the lattice
        if isinstance(p, Iv):
            if p.hi - p.lo > 1 or p.lo % s:
                return None
        elif p.count > 0 and (p.start % s or (p.count > 1 and p.stride % s)):
            return None
    W = hi // s + 1
    tot = 0
    for q0 in range(0, N, BQ):
        pieces = []
        cap = q0 + BQ - 1
        for p in D.pieces:
            if isinstance(p, Iv):
                if p.lo <= cap:
                    pieces.append(Iv(p.lo // s, p.lo // s + 1))
            else:
                n = min(p.count, (cap - p.start) // p.stride + 1)
                if n > 0:
                    pieces.append(Ap(p.start // s, p.stride // s, n))
        if pieces:
            tot += BQ * billed_cols(pieces, A, W)
    return tot


def cost_of(name, D, N, BQ, A):
    if name == "identity":
        return D.cost(N, BQ, A)
    if name == "shear":
        return cost_shear(D, N, BQ, A)
    if name.startswith("stridefold-"):
        return cost_stridefold(D, N, BQ, A, int(name.split("-")[1]))
    if name.startswith("residue-perm-"):
        return cost_residue_perm(D, N, BQ, A, int(name.split("-")[-1]))
    raise KeyError(name)


def select(D, N, BQ, A, candidates=CANDIDATES, free_only=False):
    """argmin over candidates. Ties break toward the free (class A) option."""
    best = None
    for nm in candidates:
        if free_only and (nm == "shear" or nm.startswith("stridefold")):
            continue
        c = cost_of(nm, D, N, BQ, A)
        if c is None:
            continue
        key = (c, 0 if (nm == "identity" or nm.startswith("residue")) else 1, nm)
        if best is None or key < best[0]:
            best = (key, nm, c)
    return (best[1], best[2]) if best else (None, None)
