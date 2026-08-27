"""Timing. Medians of CUDA-event timings, warmed up, cache-flushed between reps.

Deliberately conservative: a claim that survives this is not surviving a
favourable measurement setup.
"""
import statistics

import torch


def time_ms(fn, warmup=25, reps=100, flush_l2=True):
    cache = torch.empty(int(64e6 // 4), dtype=torch.int, device="cuda") if flush_l2 else None
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(reps):
        if cache is not None:
            cache.zero_()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out), statistics.stdev(out) if len(out) > 1 else 0.0


def report(rows, cols, title=None, note=None):
    if title:
        print(f"\n{title}")
        print("-" * max(len(title), sum(w for _, w in cols)))
    print("".join(f"{h:>{w}}" for h, w in cols))
    for r in rows:
        print("".join(
            (f"{v:>{w}.{3 if isinstance(v, float) else 0}f}" if isinstance(v, (int, float))
             else f"{str(v):>{w}}")
            for v, (_, w) in zip(r, cols)))
    if note:
        print(f"\n{note}")
