"""Control: how much does the selector add over 'always identity'?

2f's critique of NOTES sec 5f: agreement on non-invariant masks is not measuring
selection quality, it is measuring how often identity happens to be optimal.
Correct by construction for those cells -- my selector RETURNS identity there, so
it IS the trivial selector on that subset. The open question is the headline
number: what does always-identity score overall?

One oracle pass, both selectors scored against it.
"""
import numpy as np

from polyattn import selector, selector_oracle as so

NS = (1024, 1536, 2048)
TILES = [(128, 128), (128, 32), (128, 16), (64, 64), (64, 16), (32, 32), (16, 16)]

rows = []
for m, N, BQ, A in so.instances(NS, TILES):
    best, costs = so.oracle(m, N, BQ, A)
    pick = selector.select(m, N, BQ, A)
    inv = selector.offsets_of(m, N) is not None
    rows.append((costs, best, pick, inv, getattr(m, "family", "?")))

def score(get_pick, sub=None):
    rs = [r for r in rows if sub is None or sub(r)]
    a = np.mean([get_pick(r) == r[1] for r in rs])
    reg = np.array([r[0].get(get_pick(r), r[0]["identity"]) / r[0][r[1]] for r in rs])
    return len(rs), a, reg.mean(), reg.max()

print(f"{'selector':<22}{'cells':>7}{'agree':>9}{'mean reg':>10}{'max reg':>9}")
print("-" * 57)
for label, f, sub in (
        ("ours (all)",              lambda r: r[2], None),
        ("always-identity (all)",   lambda r: "identity", None),
        ("ours (invariant)",        lambda r: r[2], lambda r: r[3]),
        ("always-identity (inv)",   lambda r: "identity", lambda r: r[3]),
        ("ours (NOT invariant)",    lambda r: r[2], lambda r: not r[3]),
        ("always-identity (n-inv)", lambda r: "identity", lambda r: not r[3])):
    n, a, mr, xr = score(f, sub)
    print(f"{label:<22}{n:>7}{a*100:>8.1f}%{mr:>10.4f}{xr:>9.4f}")

print("\nHow often is identity the oracle's own answer?")
for lbl, sub in (("overall", None), ("invariant", lambda r: r[3]),
                 ("NOT invariant", lambda r: not r[3])):
    rs = [r for r in rows if sub is None or sub(r)]
    print(f"  {lbl:<16}{np.mean([r[1] == 'identity' for r in rs])*100:5.1f}%"
          f"  ({len(rs)} cells)")
