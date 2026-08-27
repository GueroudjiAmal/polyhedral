"""EXPERIMENT 4 -- NOT IN THE DEFAULT RUN. Its outcome space collapsed.

This was "the baseline sweep". Sub-128 BlockMask is not reachable through any API
found in torch 2.6.0 -- kernel_options BLOCK_M/BLOCK_N are dropped before the
lowering -- so the sweep has exactly ONE reachable point, which exp0 already
measures. An experiment with one point in its outcome space is not a sweep.

Kept as a REGRESSION PROBE: run it by hand against a newer torch. If sub-128
starts compiling, the baseline becomes tunable and every speedup in this project
needs requoting against a tuned opponent.

Original header follows.

EXPERIMENT 4 -- against the incumbent, at a tile size it can actually use.

docs/NOTES.md sec 3a-bis: comparing only against FlexAttention's 128x128 default
was picking the weakest opponent, and Binary Block Masking already runs at 128x32.
FlexAttention's BLOCK_SIZE is tunable, so the honest baseline is the best
FlexAttention can do, not the default.

Also worth knowing on this box: whether create_block_mask even accepts small
block sizes, and what they cost. The PyTorch docs show BLOCK_SIZE being raised
for memory, never lowered.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu                        # noqa: E402
from triton_attn import block_sparse_attention             # noqa: E402

N, D, B, H = 4096, 64, 1, 8
NAMES = ["causal", "window-128", "dilated-8", "local256+str8"]
FLEX_BLOCKS = [128, 64, 32]
OURS = [(128, 128), (128, 32), (16, 16)]


def _why(e, n=3):
    """Root cause, not just the wrapper class. BackendCompilerFailed on its own
    says nothing -- two runs reported it and neither told us why."""
    cur, out = e, []
    for _ in range(6):
        first = (str(cur).strip().splitlines() or [""])[0]
        out.append(f"{type(cur).__name__}: {first[:90]}" if first else type(cur).__name__)
        nxt = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return " <- ".join(out[:n])


def main():
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception as e:
        print(f"flex_attention unavailable: {e}")
        return
    torch.manual_seed(0)
    q4, k4, v4 = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
                  for _ in range(3))
    q3 = q4.reshape(B * H, N, D).contiguous()
    k3 = k4.reshape(B * H, N, D).contiguous()
    v3 = v4.reshape(B * H, N, D).contiguous()

    for name in NAMES:
        mod = masks_gpu.flex_mask_mod(name)
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        rows = []
        for bs in FLEX_BLOCKS:
            try:
                # fresh compile per config -- see exp0. A reused callable
                # measured the same baseline 23% apart across two jobs.
                torch._dynamo.reset()
                fa = torch.compile(flex_attention, dynamic=False)
                bm = create_block_mask(mod, B=None, H=None, Q_LEN=N, KV_LEN=N,
                                       BLOCK_SIZE=bs, _compile=True)
                kopt = {} if bs >= 128 else {"kernel_options": {"BLOCK_M": bs,
                                                                "BLOCK_N": bs}}
                ms = bench.time_ms(lambda: fa(q4, k4, v4, block_mask=bm, **kopt),
                                   warmup=25, reps=100)[0]
                rows.append([f"flex {bs}x{bs}", float("nan"), ms])
            except Exception as e:
                rows.append([f"flex {bs}x{bs}", float("nan"), f"FAIL {_why(e)}"])
        for BQ, A in OURS:
            bi = blockindex.build(m, N, BQ, A)
            idx = blockindex.to_cuda(bi)
            ms = bench.time_ms(lambda: block_sparse_attention(
                q3, k3, v3, *idx, kind, p0, p1, p2, BQ, A), reps=50)[0]
            rows.append([f"ours {BQ}x{A}", bi.waste, ms])
        bench.report(rows, [("impl", 14), ("waste", 9), ("ms", 12)],
                     title=f"{name}   N={N} B={B} H={H} D={D}",
                     note="Our kernel is a research kernel and FlexAttention is tuned;\n"
                          "read the SHAPE of the tile-size trend, not the absolute gap.")


if __name__ == "__main__":
    main()
