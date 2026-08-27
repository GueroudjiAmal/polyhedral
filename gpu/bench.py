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


def paired_time(fn_a, fn_b, warmup=25, reps=200, flush_l2=True):
    """Time two callables INTERLEAVED, and report the paired difference.

    Sequential A-then-B measurement confounds the A-vs-B difference with any
    drift over the measurement window -- clocks, thermals, another job's
    interference. Within one process that drift is small but it is not zero, and
    exp8's tightest cell has a 5.9% margin against a 2% within-job CV.

    Alternating the two and differencing per pair cancels drift to first order:
    each A sample sits between two B samples. Returns
    (median_a, median_b, median_of_per_pair_ratios, stdev_of_ratios), and the
    RATIO statistic is the one to trust -- it is paired, the medians are not.
    """
    cache = torch.empty(int(64e6 // 4), dtype=torch.int, device="cuda") if flush_l2 else None
    for _ in range(warmup):
        fn_a(); fn_b()
    torch.cuda.synchronize()

    a_s, b_s, ratios = [], [], []
    for _ in range(reps):
        pair = []
        for fn in (fn_a, fn_b):
            if cache is not None:
                cache.zero_()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            pair.append(s.elapsed_time(e))
        a_s.append(pair[0]); b_s.append(pair[1])
        if pair[0] > 0:
            ratios.append(pair[1] / pair[0])
    return (statistics.median(a_s), statistics.median(b_s),
            statistics.median(ratios),
            statistics.stdev(ratios) if len(ratios) > 1 else 0.0)
