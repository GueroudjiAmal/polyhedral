"""DIAGNOSTIC -- why does FlexAttention fail below BLOCK_SIZE 128, and what fixes it.

Two runs reported `FAIL BackendCompilerFailed` at 64x64 and 32x32 and nothing
else, because the experiment captured `type(e).__name__` and threw the message
away. So the cause has never actually been seen. That is the first thing fixed
here: the root cause is unwrapped and printed.

THE LIKELY MECHANISM, stated as a hypothesis so the output can refute it.
FlexAttention lowers to a Triton template whose tile sizes (BLOCK_M, BLOCK_N)
come from its own autotuning and default well above 32. The BlockMask's
BLOCK_SIZE and the kernel's tile have to be compatible -- the template checks
divisibility -- so asking for a 32x32 BlockMask while the kernel picks a
128-wide tile is inconsistent and the lowering fails.

If that is right, the fix is to tell the kernel to use tiles matching the mask,
via `kernel_options={"BLOCK_M": bs, "BLOCK_N": bs}`. This script tries that and
three other variations, per block size, and reports which combinations actually
compile and what they cost.

WHY IT MATTERS: FlexAttention at its 128 default is the weakest available
opponent, and every speedup this project quotes against it is an UPPER BOUND
until it is swept. 2.28x against an untuned baseline is not the same claim as
2.28x against a tuned one.
"""
import sys
import traceback

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import bench, masks_gpu                                     # noqa: E402

N, D, B, H = 4096, 64, 1, 8
NAMES = ["causal", "dilated-8"]
BLOCKS = (128, 64, 32, 16)


def root_cause(e, depth=6):
    """Unwrap BackendCompilerFailed to the exception that actually happened."""
    seen, cur = [], e
    for _ in range(depth):
        seen.append(f"{type(cur).__name__}: {str(cur).strip().splitlines()[0][:300]}"
                    if str(cur).strip() else type(cur).__name__)
        nxt = getattr(cur, "inner_exception", None) or cur.__cause__ or cur.__context__
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return seen


def strategies(bs):
    """(label, create_block_mask kwargs, flex_attention kwargs)."""
    return [
        ("default", dict(BLOCK_SIZE=bs, _compile=True), {}),
        ("no _compile", dict(BLOCK_SIZE=bs), {}),
        ("kernel_options matched", dict(BLOCK_SIZE=bs, _compile=True),
         dict(kernel_options={"BLOCK_M": bs, "BLOCK_N": bs})),
        ("kernel_options, fwd+bwd, no autotune", dict(BLOCK_SIZE=bs, _compile=True),
         dict(kernel_options={"BLOCK_M": bs, "BLOCK_N": bs,
                              "BLOCK_M1": bs, "BLOCK_N1": bs,
                              "BLOCK_M2": bs, "BLOCK_N2": bs,
                              "FORCE_USE_FLEX_ATTENTION": True})),
        ("kernel_options matched + bwd tiles", dict(BLOCK_SIZE=bs, _compile=True),
         dict(kernel_options={"BLOCK_M": bs, "BLOCK_N": bs,
                              "BLOCK_M1": bs, "BLOCK_N1": bs,
                              "BLOCK_M2": bs, "BLOCK_N2": bs})),
    ]


def main():
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception as e:
        print(f"flex_attention unavailable: {e}")
        return 1
    print(f"torch {torch.__version__}   N={N} B={B} H={H} D={D}\n")
    print("""NOTE: a SINGLE torch.compile'd callable is reused across calls in
exp0/exp4, and kernel_options may be baked in at first compile -- so passing
different kernel_options per call can be silently ignored. That is the leading
suspect for why the matched-tiles fix landed in exp0 and the 64x64 case still
failed with the same divisibility error. Each strategy below gets a FRESH
compile, and torch._dynamo.reset() runs between them, so the option cannot be
shadowed by a cached graph.\n""")
    q, k, v = (torch.randn(B, H, N, D, device="cuda", dtype=torch.float16)
               for _ in range(3))

    for name in NAMES:
        mod = masks_gpu.flex_mask_mod(name)
        print(f"=== {name} " + "=" * 56)
        for bs in BLOCKS:
            for label, bm_kw, fa_kw in strategies(bs):
                tag = f"  {bs:>4}x{bs:<4} {label:<34}"
                try:
                    bm = create_block_mask(mod, B=None, H=None, Q_LEN=N,
                                           KV_LEN=N, **bm_kw)
                except Exception as e:
                    print(tag + "create_block_mask FAILED")
                    for line in root_cause(e):
                        print(f"        {line}")
                    continue
                try:
                    torch._dynamo.reset()          # no cached graph may survive
                    fa = torch.compile(flex_attention, dynamic=False)
                    ms = bench.time_ms(lambda: fa(q, k, v, block_mask=bm, **fa_kw),
                                       warmup=10, reps=30)[0]
                    print(tag + f"OK   {ms:.4f} ms")
                except Exception as e:
                    print(tag + "flex_attention FAILED")
                    for line in root_cause(e):
                        print(f"        {line}")
                    if bs == BLOCKS[1] and label == "default":
                        print("        --- full traceback, once, for the first failure ---")
                        print("        " + "\n        ".join(
                            traceback.format_exc().strip().splitlines()[-12:]))
        print()
    print("""If 'kernel_options matched' compiles where 'default' does not, the
divisibility hypothesis is confirmed and exp0/exp4 should pass kernel_options.
If everything below 128 fails identically regardless, the limit is structural in
this torch version and the writeup must say the baseline could not be tuned --
which keeps every quoted speedup an explicit upper bound.""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
