"""Shared launch-config sweep and its boundary check.

This lived in exp0 and was CALLED from exp8, which never had the definition --
the insertion targeted `def best_launch(`, a function only exp0 has. exp8 died on
a NameError that syntax-checked clean. One definition, imported by both, so the
two cannot drift apart again.

The sweep was widened after run 1, where BOTH 16x16 rows picked w2/s2 -- the
minimum of both parameters then offered. A winner on the boundary means the
optimum may lie outside the sweep, and both headline numbers came from those
rows. Widening moved the small-tile penalty 1.61 -> 1.27, so a third of what had
been recorded as a hardware property was under-tuning.
"""

LAUNCH = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3), (2, 4),
          (4, 1), (4, 2), (4, 3), (8, 2), (8, 3))

#: num_warps cannot go below 1, so a winner at w1 is a HARDWARE FLOOR, not a
#: sweep that was too narrow. num_stages CAN go to 1. Run 2 reported "ON SWEEP
#: BOUNDARY (warps)" for both 16x16 rows -- a false alarm -- while the real gap
#: (stages) went unflagged on the 128x128 row.
_HARD_FLOOR = {"warps": 1, "stages": 1}


def _warn_if_on_boundary(cfg, label=""):
    """A winner at the edge of the sweep is a sweep that was too narrow."""
    if not cfg or "None" in cfg:
        # no config succeeded; the caller reports why. Do not turn a missing
        # measurement into a parse error two frames away.
        return "  <-- NO LAUNCH CONFIG SUCCEEDED"
    w, st = (int(x[1:]) for x in cfg.split("/"))
    ws = sorted({c[0] for c in LAUNCH})
    ss = sorted({c[1] for c in LAUNCH})
    edge = []
    for n, v, lo, hi in (("warps", w, ws[0], ws[-1]), ("stages", st, ss[0], ss[-1])):
        if v == hi:
            edge.append(f"{n} at sweep max")
        elif v == lo and lo > _HARD_FLOOR[n]:
            edge.append(f"{n} at sweep min, floor is {_HARD_FLOOR[n]}")
    return f"  <-- SWEEP TOO NARROW ({'; '.join(edge)}){label}" if edge else ""
