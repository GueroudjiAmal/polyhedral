"""PRE-REGISTRATION. Run on CPU, commit the output, THEN run the GPU jobs.

Three sessions built independent cost models. Writing each one's predictions down
before any timing turns a single wall-clock number into a test of three models
rather than one data point -- and removes the option of deciding afterwards which
prediction the hardware was really confirming.

Emits, for every configuration the gpu/ experiments will run:
  * predicted elements computed
  * predicted speedup over the 128x128 identity baseline
  * predicted argmin transform

Usage:
    .venv/bin/python gpu/predict.py > results/predictions-d4.txt
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import masks_gpu                                            # noqa: E402

from polyattn import selector                               # noqa: E402

EXP1 = dict(names=["window-128", "window-512", "local256+str8", "causal"], N=4096,
            tiles=[(128, 128), (128, 64), (128, 32), (128, 16), (64, 128), (64, 64),
                   (64, 16), (32, 128), (32, 32), (16, 128), (16, 32), (16, 16)])
EXP2 = dict(cases=[("dilated-4", 4), ("dilated-8", 8), ("local256+str8", 8)], N=4096,
            tiles=[(128, 128), (128, 32), (64, 64), (32, 32), (16, 16)])
EXP3 = dict(names=["local256+str8", "dilated-8", "window-128"], N=4096,
            tiles=[(128, 128), (128, 32), (128, 16), (64, 64), (64, 16),
                   (32, 32), (16, 16)])
EXP0 = dict(name="dilated-8", s=8, N=4096, tiles=[(128, 128), (16, 16)])


def _cost(name, N, BQ, A, transform=None):
    m = masks_gpu.numpy_mask(name)
    c = selector.costs(m, N, BQ, A)
    if not c:
        return None
    return c.get(transform) if transform else c


def main():
    out = {"session": "d4", "model": "polyattn.selector (exact closed form)",
           "note": "diagonally-invariant masks only; non-invariant -> declines",
           "exp0": {}, "exp1": {}, "exp2": {}, "exp3": {}}

    print("=" * 78)
    print("PREDICTIONS -- session d4 -- written BEFORE any GPU run")
    print("=" * 78)

    n, s, N = EXP0["name"], EXP0["s"], EXP0["N"]
    base = _cost(n, N, 128, 128, "identity")
    perm = _cost(n, N, 16, 16, f"residue-perm-{s}")
    ident16 = _cost(n, N, 16, 16, "identity")
    print(f"\n### exp0 GO/NO-GO  {n} + residue-perm-{s}  N={N}")
    print(f"  identity 128x128 elements     {base:,}")
    print(f"  identity 16x16   elements     {ident16:,}   speedup {base/ident16:.3f}x")
    print(f"  residue-perm-{s} 16x16 elements {perm:,}   speedup {base/perm:.3f}x")
    print(f"  --> PREDICTED WALL-CLOCK SPEEDUP OF THE TRANSFORM: {ident16/perm:.3f}x")
    print("      (vs the 16x16 identity kernel, which isolates the transform)")
    out["exp0"] = dict(identity_128=base, identity_16=ident16,
                       permuted_16=perm, transform_speedup=ident16 / perm)

    print(f"\n### exp1 tile shape  N={EXP1['N']}   predicted elements / speedup vs 128x128")
    for nm in EXP1["names"]:
        b = _cost(nm, EXP1["N"], 128, 128, "identity")
        row = {}
        if b is None:
            print(f"  {nm:<16} DECLINES (not diagonally invariant)")
            continue
        cells = []
        for BQ, A in EXP1["tiles"]:
            c = _cost(nm, EXP1["N"], BQ, A, "identity")
            row[f"{BQ}x{A}"] = c
            cells.append(f"{BQ}x{A}:{b/c:.2f}x")
        print(f"  {nm:<16} " + "  ".join(cells))
        out["exp1"][nm] = row

    print(f"\n### exp2 class A  N={EXP2['N']}   predicted element gain from the permutation")
    for nm, s_ in EXP2["cases"]:
        cells, row = [], {}
        for BQ, A in EXP2["tiles"]:
            i = _cost(nm, EXP2["N"], BQ, A, "identity")
            p = _cost(nm, EXP2["N"], BQ, A, f"residue-perm-{s_}")
            if i and p:
                cells.append(f"{BQ}x{A}:{i/p:.2f}x")
                row[f"{BQ}x{A}"] = dict(identity=i, permuted=p, gain=i / p)
        print(f"  {nm:<16} rp{s_}  " + "  ".join(cells))
        out["exp2"][nm] = row

    print(f"\n### exp3 selection  N={EXP3['N']}   PREDICTED ARGMIN per tile shape")
    print("  (exp3 measures the wall-clock argmin; a mismatch with this row")
    print("   falsifies the cost model as a selector -- see NOTES sec 5b/5e)")
    for nm in EXP3["names"]:
        m = masks_gpu.numpy_mask(nm)
        picks, row = [], {}
        for BQ, A in EXP3["tiles"]:
            p = selector.select(m, EXP3["N"], BQ, A)
            picks.append(f"{BQ}x{A}:{p.replace('residue-perm-', 'rp')}")
            row[f"{BQ}x{A}"] = p
        print(f"  {nm:<16} " + "  ".join(picks))
        out["exp3"][nm] = row

    print("\n" + "=" * 78)
    print("PRE-REGISTERED EXPECTATIONS -- what I expect to be WRONG, and why")
    print("=" * 78)
    print("""
Recorded now so it cannot be explained afterwards.

1. exp3 SHOULD SHOW DISAGREEMENT ON window-128, AND MY MODEL SHOULD LOSE.
   The argmin above says `shear` at every tile shape. shear is class B: NOTES
   sec 4 measured it buying 1.12x fewer elements while raising kv rows per tile
   from 16 to 31, about 1.94x traffic. My cost model counts elements and has NO
   TRAFFIC TERM, so it cannot see that. If the hardware picks identity over
   shear, the model is wrong in exactly the way it is already known to be wrong,
   and the fix is a traffic term rather than a new mechanism.
   A disagreement here is EXPECTED. A disagreement on dilated-8 + residue-perm-8
   would not be, and would be the serious result.

2. SAME FOR dilated-8, WHERE THE ARGMIN SAYS stridefold-8.
   stridefold and residue-perm reach identical element counts (both 8x); ties
   break by candidate order, which puts stridefold first. They are NOT
   equivalent: stridefold is class B and needs ~136 kv rows per 16x16 tile
   against residue-perm's 16. The exp0/exp2 comparison uses residue-perm
   explicitly rather than the argmin, so the go/no-go is unaffected -- but it
   means my selector, as scored on elements alone, would ship the wrong one.
   That is a tie-breaking defect the element-count oracle cannot penalise.

3. exp1 SHOULD SHOW WALL-CLOCK FALLING SHORT OF THE PREDICTED 1.78x ON
   window-128 AT 16x16. The prediction needs BOTH tile axes at 16, and small
   query tiles cost occupancy, MMA efficiency and per-tile softmax statistics --
   none of which the model prices (NOTES sec 3a). If measured tracks predicted
   here, sec 3a's central caveat was wrong and mechanism 2 is worth revisiting.

4. exp2's dilated-8 PREDICTION OF 7.79x IS THE ONE I EXPECT TO HOLD, because
   the transform is class A: the permutation is applied to K/V once and every
   tile stays rectangular and contiguous, so there is no traffic term to omit.
   If this one misses badly, the class A / class B distinction itself is wrong.
""")
    print("=" * 78)
    print("JSON follows for machine comparison against the other two sessions.")
    print("=" * 78)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
