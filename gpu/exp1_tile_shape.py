"""EXPERIMENT 1 -- the one that decides mechanism 2. docs/NOTES.md sec 3a.

CLAIM UNDER TEST: for a single-interval mask the element-count win needs BOTH
tile axes shrunk (window-128: 2.00 at 128x128, 128x16 AND 16x128; 1.12 only at
16x16), and shrinking the QUERY axis is the expensive one -- occupancy, MMA
efficiency, per-tile softmax statistics -- none of which the analytical model
charges for.

WHAT FALSIFIES WHAT:
  * wall-clock tracks elements down to 16x16  -> the model is predictive; the
    1.78x is real and mechanism 2 comes back off the shelf.
  * wall-clock flattens or REGRESSES as BQ falls while elements keep dropping
    -> the model overprices the query axis, sec 3a's suspicion is confirmed, and
    the 'unpriced' caveat becomes a measured cost.
Either way this is the first number in the project that is not an element count.
"""
import sys

import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu                        # noqa: E402
from triton_attn import block_sparse_attention             # noqa: E402

N, D, BH = 4096, 64, 8
TILES = [(128, 128), (128, 64), (128, 32), (128, 16),
         (64, 128), (64, 64), (64, 16),
         (32, 128), (32, 32), (16, 128), (16, 32), (16, 16)]
NAMES = ["window-128", "window-512", "sinks-like:local256+str8", "causal"]
NAMES = ["window-128", "window-512", "local256+str8", "causal"]


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    for name in NAMES:
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        rows, base_ms, base_el = [], None, None
        for BQ, A in TILES:
            bi = blockindex.build(m, N, BQ, A)
            kvi, kvn, kvp = blockindex.to_cuda(bi)
            fn = lambda: block_sparse_attention(q, k, v, kvi, kvn, kvp,
                                                kind, p0, p1, p2, BQ, A)
            try:
                ms, sd = bench.time_ms(fn)
            except Exception as e:                 # OOR tile shape / resource limits
                rows.append([f"{BQ}x{A}", bi.waste, float("nan"), float("nan"),
                             f"SKIP {type(e).__name__}"])
                continue
            if base_ms is None:
                base_ms, base_el = ms, bi.elements
            rows.append([f"{BQ}x{A}", bi.waste, ms,
                         base_el / bi.elements, base_ms / ms])
        bench.report(
            rows,
            [("tile", 10), ("waste", 9), ("ms", 10), ("elem gain", 11), ("time gain", 11)],
            title=f"{name}   N={N} BH={BH} D={D}   baseline = 128x128",
            note="elem gain = what the cost model predicts. time gain = what the GPU did.\n"
                 "A gap between the two columns IS the result.")


if __name__ == "__main__":
    main()
