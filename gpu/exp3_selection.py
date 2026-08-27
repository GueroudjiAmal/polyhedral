"""EXPERIMENT 3 -- the top-ranked contribution. docs/NOTES.md sec 5b/5e.

CLAIM UNDER TEST: transform selection is a function of (predicate, BQ, A), and a
cost model can pick the winner. Analytically the argmin for local256+str8 moves
identity -> rp2 -> rp4 -> rp8 across tile shapes, and sec 5e showed it even splits
WITHIN a max(BQ,A) class.

This asks the only question that matters for the compiler framing:
DOES THE WALL-CLOCK ARGMIN MATCH THE ELEMENT-COUNT ARGMIN?

  * agrees everywhere -> the cost model is a valid selector; the contribution
    stands and the compiler has something to optimise against.
  * disagrees -> the cost model selects the wrong transform on real hardware,
    and item 1 of the novelty ranking is dead in its current form. That would be
    the single most important negative result the project could produce, so it
    is worth more care than anything else here.

Reports both argmins per tile shape and flags every disagreement.

THE DISAGREEMENT COUNT IS A PROPERTY OF THE (MASK, CANDIDATE SET) PAIR, NOT OF
THE MASK. Measured analytically with the mask and criteria held fixed and only
the candidate list varied, local256+str8 runs 0% / 25% / 38% / 38% / 0% across
{id,rp2} / {id,rp2,rp4} / {id,rp2,rp4,rp8} / {id..rp16} / {id,rp3,rp5} -- rp3 and
rp5 do not collapse a stride-8 lattice, so a set containing only those has
nothing to disagree about.

Consequences, both of which apply to whatever number this prints:
  * the candidate set must be quoted WITH the number, and must be the same set
    the selectors use, or the wall-clock result is not comparable to the counting
    result it exists to adjudicate;
  * widening the set can only ADD disagreements, so a small count from a narrow
    set is not evidence of a small effect. The set below is the widest the kernel
    can run, to make the number an upper-ish bound rather than an arbitrary point.
"""
import sys

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import blockindex, bench, masks_gpu, permute               # noqa: E402
from triton_attn import block_sparse_attention             # noqa: E402

N, D, BH = 4096, 64, 8
NAMES = ["local256+str8", "dilated-8", "window-128"]
TILES = [(128, 128), (128, 32), (128, 16), (64, 64), (64, 16), (32, 32), (16, 16)]
PERMS = [None, 2, 4, 8, 16]      # {identity, rp2, rp4, rp8, rp16} -- QUOTE THIS
                                 # WITH ANY NUMBER THIS EXPERIMENT PRODUCES


class _Dense:
    def __init__(self, M):
        self.M = M
    def row_cols(self, q, N):
        return self.M[q]


def main():
    torch.manual_seed(0)
    q, k, v = (torch.randn(BH, N, D, device="cuda", dtype=torch.float16) for _ in range(3))
    disagreements = 0
    skipped = []
    total_cells = 0
    for name in NAMES:
        m = masks_gpu.numpy_mask(name)
        kind, p0, p1, p2 = masks_gpu.triton_params(name)
        M = np.stack([m.row_cols(i, N) for i in range(N)])

        variants = {}
        for s in PERMS:
            tag = "identity" if s is None else f"rp{s}"
            p_np = permute.identity_perm(N) if s is None else permute.residue_perm(N, s)
            perm = torch.from_numpy(p_np).cuda()
            variants[tag] = (M[p_np][:, p_np], perm,
                             tuple(permute.apply_perm(x, perm) for x in (q, k, v)))

        rows = []
        for BQ, A in TILES:
            el, ms = {}, {}
            for tag, (Mv, perm, (qv, kv_, vv)) in variants.items():
                bi = blockindex.build(_Dense(Mv), N, BQ, A)
                idx = blockindex.to_cuda(bi)
                el[tag] = bi.elements
                ms[tag] = bench.time_ms(lambda: block_sparse_attention(
                    qv, kv_, vv, *idx, kind, p0, p1, p2, BQ, A,
                    perm_q=perm.int(), perm_kv=perm.int()), reps=60)[0]
            # PRECONDITION. A cell where the candidates are within `margin` on
            # element count cannot exhibit a counting-vs-hardware disagreement at
            # all, so counting it as "agreement" is counting a cell that could
            # not have disagreed. Without this, a clean zero is ambiguous between
            # (i) hardware agrees with counting, (ii) the candidate set was too
            # narrow, (iii) there was nothing to disagree about -- and only (i) is
            # the answer anyone would draw.
            lo2 = sorted(el.values())[:2]
            if len(lo2) < 2 or lo2[0] == 0 or (lo2[1] - lo2[0]) / lo2[0] < 0.02:
                skipped.append(f"{name} {BQ}x{A}")
                continue
            a_el = min(el, key=el.get)
            a_ms = min(ms, key=ms.get)
            agree = a_el == a_ms
            disagreements += (not agree)
            total_cells += 1
            rows.append([f"{BQ}x{A}", a_el, a_ms, "yes" if agree else "NO",
                         ms[a_ms] / ms[a_el] if not agree else 1.0])
        bench.report(
            rows,
            [("tile", 10), ("argmin elems", 14), ("argmin time", 13),
             ("agree", 8), ("cost of", 10)],
            title=f"{name}   N={N} BH={BH}   selector validity",
            note="'cost of' = how much slower the model's pick is than the true best.\n"
                 "1.00 means no loss; anything above it is the price of a wrong selection.")
    cset = "{" + ", ".join("identity" if p is None else f"rp{p}" for p in PERMS) + "}"
    print(f"\n{'=' * 60}")
    print(f"CANDIDATE SET: {cset}")
    print("  The count below is for THIS set. The same masks under {identity, rp2,")
    print("  rp4} can give 0% where this gives 38%. Widening only adds")
    print("  disagreements, so a small number here is not a small effect.")
    if skipped:
        print(f"SKIPPED {len(skipped)} cell(s) with <2% element-count separation --")
        print("  no counting-level disagreement was possible, so they cannot")
        print("  contribute evidence either way: " + ", ".join(skipped[:6])
              + (" ..." if len(skipped) > 6 else ""))
    print(f"TOTAL DISAGREEMENTS: {disagreements}  (over {total_cells} scored cells)")
    print("Zero -> the element-count cost model is a valid transform selector.")
    print("Nonzero -> it is not, and NOTES novelty item 1 needs restating.")


if __name__ == "__main__":
    main()
