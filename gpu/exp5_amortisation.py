"""EXPERIMENT 5 -- the fork the headline sits on. docs/NOTES.md §4.

exp0 measured the permutation at 0.105 ms against a 0.122 ms kernel. That is not
a rounding error, it is the whole result:

    permute PER LAYER          0.227 ms  ->  1.41x vs FlexAttention default
    permute ONCE PER FORWARD   0.122 ms  ->  2.61x

§4 claims class A is "free" because the permutation is applied once per layer and
amortised. §5 lists that as ARGUED, NOT MEASURED. It is the load-bearing
assumption of mechanism 1 and it has never been tested.

The argument for once-per-forward: every other op in a transformer block is
per-token and therefore permutation-equivariant, so you can permute at the
embedding and invert at the output, running the whole stack in permuted space.
If that holds, the permutation cost divides by the layer count and vanishes.

This measures all three regimes over a realistic depth sweep, and checks
CORRECTNESS of the once-per-forward path rather than assuming equivariance --
because "every other op is per-token" is exactly the kind of claim this project
has learned not to accept unverified.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute, reference     # noqa: E402
from triton_attn import block_sparse_attention              # noqa: E402

NAME, STRIDE = "dilated-8", 8
N, D, BH = 4096, 64, 8
LAYERS = (1, 4, 8, 16, 32)


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def _rope(x, pos):
    """Position-dependent, NOT elementwise. This is the silent-failure case.

    RoPE does not break permutation-equivariance -- it REQUIRES that position ids
    travel with the tokens. Permute the tokens and regenerate positions with
    arange() and you get a wrong answer that still looks numerically plausible,
    which is why exp5 exercises it rather than testing only elementwise ops.
    """
    d = x.shape[-1] // 2
    freq = torch.exp(-np.log(10000.0) * torch.arange(d, device=x.device) / d)
    ang = pos[:, None].float() * freq[None, :]
    c, s_ = torch.cos(ang).half(), torch.sin(ang).half()
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * c - b * s_, a * s_ + b * c], dim=-1)


def _mlp_like(x):
    """Elementwise, so permutation-equivariant by construction."""
    return torch.nn.functional.gelu(x) * 0.5 + x


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3))
    m = masks_gpu.numpy_mask(NAME)
    kind, p0, p1, p2 = masks_gpu.triton_params(NAME)
    M = np.stack([m.row_cols(i, N) for i in range(N)])
    p_np = permute.residue_perm(N, STRIDE)
    perm = torch.from_numpy(p_np).cuda()
    inv = permute.invert(perm)
    Mp = M[p_np][:, p_np]

    idx_i = blockindex.to_cuda(blockindex.build(_Dense(M), N, 128, 128))
    bi_p = blockindex.build(_Dense(Mp), N, 16, 16)
    idx_p = blockindex.to_cuda(bi_p)
    pi = perm.int()

    def attn_plain(x):
        return block_sparse_attention(x, k, v, *idx_i, kind, p0, p1, p2, 128, 128)

    def attn_perm(x, kp, vp):
        return block_sparse_attention(x, kp, vp, *idx_p, kind, p0, p1, p2, 16, 16,
                                      perm_q=pi, perm_kv=pi)

    # ---- correctness of the once-per-forward path, not assumed -----------------
    kp, vp = permute.apply_perm(k, perm), permute.apply_perm(v, perm)
    pos = torch.arange(N, device="cuda")
    x = q
    for _ in range(3):
        x = _mlp_like(_rope(attn_plain(x), pos))
    ref3 = x

    # positions must travel WITH the tokens -- this is the whole check
    pos_p = pos.index_select(0, perm.long())
    xp = permute.apply_perm(q, perm)
    for _ in range(3):
        xp = _mlp_like(_rope(attn_perm(xp, kp, vp), pos_p))
    got3 = xp.index_select(1, inv.long())
    err = reference.max_abs_err(got3, ref3)

    # and the negative control: regenerating positions must FAIL, or the test
    # above proves nothing about whether it was exercising RoPE at all
    xb = permute.apply_perm(q, perm)
    for _ in range(3):
        xb = _mlp_like(_rope(attn_perm(xb, kp, vp), pos))     # WRONG on purpose
    bad = reference.max_abs_err(xb.index_select(1, inv.long()), ref3)
    print(f"negative control (positions regenerated, not permuted): err {bad:.2e}"
          f"  {'OK -- the check has teeth' if bad > 10 * max(err, 1e-6) else 'SUSPECT'}")
    print(f"3-layer stack with RoPE, positions permuted: max abs err {err:.2e}  "
          f"{'OK -- equivariance holds' if err < 2e-2 else 'FAIL -- STOP'}")
    print("  (accumulating fp16 through 3 layers, so the tolerance is loose;")
    print("   a real break shows as orders of magnitude, not 1e-2.)")
    if err >= 2e-2:
        return 1

    rows = []
    for L in LAYERS:
        def plain():
            x = q
            for _ in range(L):
                x = _mlp_like(attn_plain(x))
            return x

        def per_layer():
            x = q
            for _ in range(L):
                xp_ = permute.apply_perm(x, perm)
                kp_ = permute.apply_perm(k, perm)
                vp_ = permute.apply_perm(v, perm)
                x = _mlp_like(attn_perm(xp_, kp_, vp_).index_select(1, inv.long()))
            return x

        def once():
            kp_ = permute.apply_perm(k, perm)
            vp_ = permute.apply_perm(v, perm)
            xp_ = permute.apply_perm(q, perm)
            for _ in range(L):
                xp_ = _mlp_like(attn_perm(xp_, kp_, vp_))
            return xp_.index_select(1, inv.long())

        t_plain = bench.time_ms(plain, warmup=5, reps=20)[0]
        t_layer = bench.time_ms(per_layer, warmup=5, reps=20)[0]
        t_once = bench.time_ms(once, warmup=5, reps=20)[0]
        rows.append([L, t_plain, t_layer, t_plain / t_layer, t_once,
                     t_plain / t_once])

    bench.report(
        rows,
        [("layers", 8), ("plain ms", 10), ("per-layer", 11), ("gain", 8),
         ("once", 10), ("gain", 8)],
        title=f"{NAME} + residue-perm-{STRIDE}   permutation amortisation   "
              f"N={N} BH={BH} D={D}",
        note="plain = 128x128 identity stack (our own baseline, NOT FlexAttention).\n"
             "per-layer = permute/invert around every attention -- the pessimistic bound.\n"
             "once = permute at the input, run the stack in permuted space, invert once --\n"
             "       what NOTES sec 4 assumes and sec 5 flags as argued rather than measured.\n"
             "If the two columns converge as depth grows, the amortisation claim holds.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
