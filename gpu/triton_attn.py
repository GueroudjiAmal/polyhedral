"""Block-sparse attention forward, with the tile shape (BQ, A) as a parameter.

The point of parameterising the tile shape is that docs/NOTES.md sec 3a claims the
element-count win on windowed masks needs BOTH tile axes small, and that the
query axis is the expensive one -- a claim the analytical model cannot price
because it charges nothing for occupancy, MMA efficiency, or per-tile softmax
statistics. This kernel is where that gets a number.

Forward only, fp16 in / fp16 out with fp32 accumulation. No backward pass: the
backward has a different access pattern and none of the claims are about it.

UNTESTED AS WRITTEN -- authored on a machine with no CUDA device. Run
gpu/test_correctness.py before believing any timing this produces.
"""
import torch
import triton
import triton.language as tl

# Triton >= 3.x refuses to read a plain Python global from inside @triton.jit:
#   "Cannot access global variable CAUSAL ... only ... annotated as constexpr"
# The annotation form keeps the runtime value a plain int, so host-side code and
# masks_gpu.py are unaffected.
CAUSAL: tl.constexpr = 0
WINDOW: tl.constexpr = 1
DILATED: tl.constexpr = 2
LOCAL_STRIDED: tl.constexpr = 3
TWOBAND: tl.constexpr = 4

# These duplicate masks_gpu.py's copy, and the docstring of _live claims the two
# mirror each other. Claiming is not checking: a silent divergence would send
# every masked tile down the wrong predicate and still produce plausible output.
def _assert_kinds_match():
    import masks_gpu as _m
    mine = dict(CAUSAL=CAUSAL, WINDOW=WINDOW, DILATED=DILATED,
                LOCAL_STRIDED=LOCAL_STRIDED, TWOBAND=TWOBAND)
    theirs = {k: getattr(_m, k) for k in mine}
    if mine != theirs:
        raise RuntimeError(f"kind constants diverged: triton_attn={mine} masks_gpu={theirs}")


@triton.jit
def _live(d, KIND: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr):
    """d = q_idx - kv_idx, elementwise. Mirrors gpu/masks_gpu.py exactly."""
    causal = d >= 0
    if KIND == CAUSAL:
        return causal
    if KIND == WINDOW:
        return causal & (d < P0)
    if KIND == DILATED:
        return causal & ((d % P0) == 0)
    if KIND == LOCAL_STRIDED:
        return causal & ((d < P0) | ((d % P1) == 0))
    return causal & ((d < P0) | ((d >= P1) & (d < P1 + P2)))


@triton.jit
def _attn_fwd(Q, K, V, O, KVI, KVN, KVP, PQ, PK,
              stride_qb, stride_qn, stride_kb, stride_kn,
              stride_vb, stride_vn, stride_ob, stride_on,
              scale, MAXKV, WSEL, stride_wb, N_KV,
              KIND: tl.constexpr, P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr,
              BQ: tl.constexpr, A: tl.constexpr, D: tl.constexpr,
              GATHER_KV: tl.constexpr, GATHER_MULT: tl.constexpr,
              GATHER_SCATTER: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BQ + tl.arange(0, BQ)
    offs_d = tl.arange(0, D)

    q = tl.load(Q + pid_b * stride_qb + offs_m[:, None] * stride_qn + offs_d[None, :])
    # Original-coordinate indices, for evaluating the mask predicate after a
    # class A permutation has physically reordered Q/K/V. Identity when there is
    # no permutation. Two vector loads per tile, not BQ*A -- negligible.
    pq = tl.load(PQ + offs_m)

    acc = tl.zeros([BQ, D], dtype=tl.float32)
    # FINITE sentinel, not -inf. A tile is visited when ANY row of the block
    # reaches it, so a row can be entirely dead in the first tile it sees. With
    # m_i = -inf that row computes exp(-inf - -inf) = exp(nan) and every
    # subsequent accumulation is poisoned; the l_i > 0 guard below catches
    # l_i == 0 but not l_i == nan, so the row would return NaN rather than zero.
    # Reachable at 128x128 on window-128, not only at small tiles.
    m_i = tl.full([BQ], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BQ], dtype=tl.float32)

    n_blocks = tl.load(KVN + pid_m)
    for j in range(0, n_blocks):
        kvb = tl.load(KVI + pid_m * MAXKV + j)
        partial = tl.load(KVP + pid_m * MAXKV + j)
        offs_n = kvb * A + tl.arange(0, A)

        # GATHER_KV isolates the ONE hardware term that differs by transform at a
        # fixed tile shape: memory access. False = K/V physically permuted, rows
        # read contiguously (class A). True = K/V left in place and rows gathered
        # per tile (what class B must do, and cannot amortise). Same tiles, same
        # element count, same output -- so any time difference IS the traffic term.
        if GATHER_KV:
            rows_n = tl.load(PK + offs_n)
        else:
            rows_n = offs_n
        k = tl.load(K + pid_b * stride_kb + rows_n[:, None] * stride_kn + offs_d[None, :])
        if GATHER_MULT > 1:
            # Touch R = A*GATHER_MULT distinct rows to produce an A-wide tile,
            # which is what class B must do (A + a*(BQ-1) rows, not A). Weights
            # come from a RUNTIME tensor [1,0,0,...] so the extra loads cannot be
            # eliminated as dead, and the arithmetic stays exact.
            w0 = tl.load(WSEL + pid_b * stride_wb + 0)
            k = k * w0
            for r in tl.static_range(1, GATHER_MULT):
                # CONTIGUOUS: extra rows follow the tile, so the row COUNT is paid
                # and coalescing is kept. SCATTERED: they come through the
                # permutation, so both are paid. The two sweeps bracket any class
                # B configuration, and the distance between them is exactly the
                # cache-residency question NOTES §5 could only speculate about.
                if GATHER_SCATTER:
                    extra = tl.load(PK + ((offs_n + r * A) % N_KV))
                else:
                    extra = (rows_n + r * A) % N_KV
                kr = tl.load(K + pid_b * stride_kb
                             + extra[:, None] * stride_kn + offs_d[None, :])
                k = k + kr * tl.load(WSEL + pid_b * stride_wb + r)
        qk = tl.dot(q, tl.trans(k)) * scale

        # The kv index vector is loaded UNCONDITIONALLY even though only partial
        # tiles use it. A tl.load inside a runtime `if` cannot be software
        # pipelined -- Triton fails with "operation scheduled before its
        # operands" -- and dropping num_stages to 1 to work around that would
        # slow every tile, which is the wrong trade in a kernel whose entire
        # purpose is a timing. This load is A int32s against a BQ*A tile, so
        # hoisting it costs essentially nothing.
        pk = tl.load(PK + offs_n)

        # BRANCH-FREE MASKING, and this is a deliberate concession.
        #
        # Ideally only PARTIAL tiles pay for the BQ*A predicate evaluation --
        # that is FlexAttention's three-way BlockMask split, and PyTorch report
        # 15-20% for masking every computed element. But Triton's software
        # pipeliner cannot schedule ANY runtime `if` in the loop body ("operation
        # scheduled before its operands"), and turning pipelining off to keep the
        # branch would slow every tile, which is worse.
        #
        # So the predicate is evaluated on every visited tile and `partial` is
        # folded into the where. WHAT THIS COSTS: our kernel forfeits the
        # full-tile mask-skip that FlexAttention keeps, so it is CONSERVATIVE
        # against that baseline -- the bias runs against our own claim, which is
        # the safe direction. It does NOT bias our own configurations against
        # each other (identity vs residue-perm, 128x128 vs 16x16) because every
        # one pays the same overhead.
        #
        # WHAT IT DOES NOT CHANGE: which tiles are visited at all. Dead tiles are
        # still skipped via kv_idx, so every element count in NOTES sec 3 still
        # describes exactly the work this kernel does.
        #
        # FOLLOW-UP if a result lands close enough for 15-20% to matter: order
        # kv_idx as [full tiles | partial tiles] and run two branch-free loops.
        # That recovers the optimisation and still pipelines. Not done here
        # because it touches blockindex and cannot be tested without a GPU.
        d = pq[:, None] - pk[None, :]
        qk = tl.where((partial == 0) | _live(d, KIND, P0, P1, P2),
                      qk, -float("inf"))

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(V + pid_b * stride_vb + rows_n[:, None] * stride_vn + offs_d[None, :])
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_new

    # a query row with no live key contributes nothing; keep it finite
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]
    tl.store(O + pid_b * stride_ob + offs_m[:, None] * stride_on + offs_d[None, :],
             acc.to(O.dtype.element_ty))


def block_sparse_attention(q, k, v, kv_idx, kv_num, kv_partial,
                           kind, p0, p1, p2, BQ, A, scale=None,
                           perm_q=None, perm_kv=None, gather_kv=False,
                           gather_mult=1, gather_scatter=True,
                           num_warps=4, num_stages=2):
    """q,k,v: [BH, N, D] contiguous, fp16. Returns [BH, N, D].

    perm_q / perm_kv: int32 [N] mapping permuted position -> ORIGINAL position.
    Pass them when q/k/v have been physically reordered by a class A permutation;
    the kernel needs them to evaluate the mask predicate, which is defined in
    original coordinates. Defaults to identity.
    """
    BH, N, D = q.shape
    if perm_q is None:
        perm_q = torch.arange(N, device=q.device, dtype=torch.int32)
    if perm_kv is None:
        perm_kv = perm_q
    # [1, 0, 0, ...] per batch-head: keeps the extra gather loads alive without
    # changing the result. Runtime values, so they cannot be constant-folded.
    wsel = torch.zeros(BH, max(gather_mult, 1), device=q.device, dtype=q.dtype)
    wsel[:, 0] = 1
    _assert_kinds_match()
    assert N % BQ == 0 and N % A == 0, "cost model and kernel both assume this"
    assert k.shape == v.shape == q.shape
    o = torch.empty_like(q)
    grid = (N // BQ, BH)

    def _launch(stages):
        _attn_fwd[grid](
            q, k, v, o, kv_idx, kv_num, kv_partial, perm_q, perm_kv,
            q.stride(0), q.stride(1), k.stride(0), k.stride(1),
            v.stride(0), v.stride(1), o.stride(0), o.stride(1),
            scale or D ** -0.5, kv_idx.shape[1],
            wsel, wsel.stride(0), N,
            KIND=kind, P0=p0, P1=p1, P2=p2, BQ=BQ, A=A, D=D, GATHER_KV=gather_kv,
            GATHER_MULT=gather_mult, GATHER_SCATTER=gather_scatter,
            num_warps=num_warps, num_stages=stages,
        )

    try:
        _launch(num_stages)
    except Exception as e:
        # The pipeliner can refuse a (BQ, A) it cannot schedule. Falling back to
        # num_stages=1 keeps the run alive, but it is NOT the same kernel: no
        # software pipelining means every timing from this configuration is
        # pessimistic and must not be compared against a pipelined one. Say so
        # loudly rather than silently producing a slower number.
        if "scheduled before its operands" not in str(e):
            raise
        key = (kind, BQ, A, num_warps)
        if key not in _PIPELINE_FALLBACK:
            _PIPELINE_FALLBACK.add(key)
            print(f"  WARNING: Triton could not pipeline kind={kind} {BQ}x{A} "
                  f"num_warps={num_warps}; retrying num_stages=1. "
                  f"TIMINGS FOR THIS CONFIG ARE NOT COMPARABLE.", flush=True)
        _launch(1)
    return o


_PIPELINE_FALLBACK = set()
