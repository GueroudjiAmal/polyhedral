"""Dense masked attention in torch. Slow, obviously correct, the oracle."""
import torch


def attention_reference(q, k, v, mask_bool, scale=None):
    """q,k,v: [BH, N, D] fp16/fp32. mask_bool: [N, N] True where live."""
    scale = scale or q.shape[-1] ** -0.5
    s = (q.float() @ k.float().transpose(-1, -2)) * scale
    s = s.masked_fill(~mask_bool[None], float("-inf"))
    p = torch.softmax(s, dim=-1)
    # rows with no live element softmax to NaN; define them as zero output
    p = torch.nan_to_num(p, nan=0.0)
    return (p @ v.float()).to(v.dtype)


def max_abs_err(a, b):
    return (a.float() - b.float()).abs().max().item()
