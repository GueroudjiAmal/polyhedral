# GPU harness — turning element counts into measurements

Everything in this repo outside `gpu/` is an exact element count. Nothing has
been measured. This directory converts each surviving claim into a wall-clock
experiment with a stated falsifier.

> **The code here has never been executed.** It was written on a machine with no
> CUDA device (verified: no `/dev/nvidia*`, no `libcuda`, Intel integrated only).
> Expect to fix things. `test_correctness.py` exists so that failures show up as
> failures rather than as plausible-looking timings.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install -r gpu/requirements.txt      # pin torch to the box's CUDA
```

## Run order — do not skip ahead

```bash
./gpu/run_all.sh              # or the steps below individually
```

| step | file | why it comes first |
|---|---|---|
| 0 | `env_check.py` | prints device, compute capability, whether FlexAttention exists |
| 1 | `test_correctness.py` | **gate.** Kernel vs dense reference at 7 tile shapes, 5 masks, plus the permuted path. Also checks the numpy mask and the torch mask are the same mask, and that `BlockIndex` agrees with `polyattn.cost` |
| 2 | `exp1_tile_shape.py` | decides mechanism 2 |
| 3 | `exp2_class_a.py` | decides mechanism 1 — the go/no-go gate |
| 4 | `exp3_selection.py` | decides the top-ranked contribution |
| 5 | `exp4_flex_baseline.py` | against the incumbent, at tile sizes it can actually use |

## What each experiment can falsify

**exp1 — tile shape** (`docs/NOTES.md` §3a). The model says window-128 wastes
2.00× at 128×128, 128×16 *and* 16×128, reaching 1.12 only at 16×16 — so the win
needs both axes small, and the query axis is the one that costs occupancy, MMA
efficiency and per-tile softmax statistics. The model charges nothing for any of
that.
- *time tracks elements down to 16×16* → model is predictive, mechanism 2 revives.
- *time flattens or regresses while elements keep falling* → §3a's suspicion is
  confirmed and "unpriced" becomes a measured cost.

**exp2 — class A** (§4). Residue permutation should cut dilated-8's computed
elements 7.79× at no traffic cost. Permutation cost is timed and reported
**separately**; amortising it over layers is left to the reader.
- *no meaningful speedup despite ~8× fewer elements* → mechanism 1 dies.

**exp3 — selection** (§5b, §5e). The whole compiler framing rests on a cost model
that picks the right transform. This reports the element-count argmin and the
wall-clock argmin per tile shape and flags every disagreement.
- *disagreements > 0* → the model selects the wrong transform on real hardware,
  and novelty item 1 is dead in its current form. **This is the most valuable
  negative result the project could produce.**

**exp4 — baseline** (§3a-bis). FlexAttention's `BLOCK_SIZE` is tunable and Binary
Block Masking already runs at 128×32, so 128×128 is the weakest available
opponent. Also establishes empirically whether small block sizes work at all here
— the PyTorch docs only ever show `BLOCK_SIZE` being *raised*.

## Design choices that keep it honest

- **One mask definition, three consumers.** numpy (`polyattn`), Triton constexpr,
  and FlexAttention `mask_mod` all derive from `masks_gpu.SPECS`, and step 1
  checks they agree rather than assuming it.
- **The kernel is driven by the same tiling the model counts.** `blockindex.build`
  produces exactly the tiles `polyattn.cost` charges for, so a timing and a
  prediction refer to the same work. Divergence is the signal.
- **Full vs partial tiles**, mirroring FlexAttention: only partial tiles pay for
  mask evaluation.
- **Permutation cost is never folded into kernel time.**
- **L2 flushed between reps**, medians not means, 25 warmup iterations.
- Forward only, fp16 with fp32 accumulation. The backward has a different access
  pattern and no claim here is about it.

## Known gaps

- No backward pass.
- The Triton kernel is not autotuned. A tuned baseline would change absolute
  numbers; read tile-size *trends*, not absolute gaps against FlexAttention.
- `blockindex.build` materialises an N×N mask — fine to N≈16384, not beyond.
- exp3 permutes Q/K/V per variant outside the timed region, which is the
  favourable-to-us choice; exp2 reports what that costs.
