"""One mask definition, three consumers: numpy (polyattn), Triton, FlexAttention.

They must agree exactly or the comparison is meaningless, so `check_agreement`
below is called by the correctness suite rather than trusted.
"""
from polyattn import masks

# Triton constexpr mask kinds. d = q_idx - kv_idx throughout.
CAUSAL, WINDOW, DILATED, LOCAL_STRIDED, TWOBAND = 0, 1, 2, 3, 4

SPECS = {
    "causal":         (CAUSAL, 0, 0, 0, lambda: masks.Causal()),
    "window-128":     (WINDOW, 128, 0, 0, lambda: masks.SlidingWindow(128)),
    "window-512":     (WINDOW, 512, 0, 0, lambda: masks.SlidingWindow(512)),
    "dilated-4":      (DILATED, 4, 0, 0, lambda: masks.Dilated(4)),
    "dilated-8":      (DILATED, 8, 0, 0, lambda: masks.Dilated(8)),
    "local256+str8":  (LOCAL_STRIDED, 256, 8, 0, lambda: masks.LocalStrided(256, 8)),
    "twoband-1024":   (TWOBAND, 128, 1024, 128, lambda: masks.TwoBand(128, 1024)),
    "twoband-1000":   (TWOBAND, 128, 1000, 128, lambda: masks.TwoBand(128, 1000)),
}


def numpy_mask(name):
    return SPECS[name][4]()


def triton_params(name):
    kind, p0, p1, p2, _ = SPECS[name]
    return kind, p0, p1, p2


def flex_mask_mod(name):
    """A mask_mod for torch.nn.attention.flex_attention.create_block_mask."""
    kind, p0, p1, p2, _ = SPECS[name]
    if kind == CAUSAL:
        return lambda b, h, q, kv: q >= kv
    if kind == WINDOW:
        return lambda b, h, q, kv: (q >= kv) & (q - kv < p0)
    if kind == DILATED:
        return lambda b, h, q, kv: (q >= kv) & ((q - kv) % p0 == 0)
    if kind == LOCAL_STRIDED:
        return lambda b, h, q, kv: (q >= kv) & (((q - kv) < p0) | ((q - kv) % p1 == 0))
    if kind == TWOBAND:
        return lambda b, h, q, kv: (q >= kv) & (((q - kv) < p0)
                                                | (((q - kv) >= p1) & ((q - kv) < p1 + p2)))
    raise ValueError(kind)


def dense_bool(name, N, device="cuda"):
    """The ground truth every implementation is checked against."""
    import torch
    q = torch.arange(N, device=device)[:, None]
    kv = torch.arange(N, device=device)[None, :]
    return flex_mask_mod(name)(None, None, q, kv)


def check_agreement(name, N=512):
    """numpy polyattn mask == torch dense mask. Run before trusting any timing."""
    import numpy as np
    import torch
    a = torch.from_numpy(np.stack([numpy_mask(name).row_cols(q, N) for q in range(N)]))
    b = dense_bool(name, N, device="cpu")
    return bool(torch.equal(a, b.cpu()))
