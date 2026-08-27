# Running the GPU harness on Polaris (ALCF)

Polaris: 4× **A100 40GB** per node (sm_80), AMD EPYC Milan host, **PBS Pro**
scheduler. Nothing here needs more than one GPU on one node — these are
single-kernel microbenchmarks, not a scaling study.

> **The kernel has never been executed anywhere.** It was written on a machine
> with no CUDA device. `job_smoke.pbs` is a gate, not a formality — expect to fix
> something on the first submission.

## Three commands

```bash
# 1. on a LOGIN node (compute nodes have no outbound network)
./polaris/setup.sh

# 2. set your allocation in both job files
sed -i 's/CHANGE_ME_PROJECT/<your_project>/' polaris/job_*.pbs

# 3. gate first, then the experiments
qsub polaris/job_gonogo.pbs         # smoke test + THE one measurement, ~10 min
qstat -u $USER
#   ... read results/gonogo-*.txt in full ...
qsub polaris/job_experiments.pbs    # only if the go/no-go says it is worth it
```

**Submit `job_gonogo.pbs` first, not the full suite.** It runs the correctness
gate and then exactly one measurement: `dilated-8 + residue-perm-8`, the case
where the analysis predicts the largest effect (7.79× fewer elements) and where
the transform is class A so the permutation is a one-time cost. If that ratio
does not appear in wall-clock, mechanism 1 is dead and the broad sweep is not
worth a queue slot.

## Polaris specifics that bite

| thing | what to do |
|---|---|
| **Login nodes have no GPU** | `nvidia-smi` fails there. All verification happens inside a job. |
| **Compute nodes have no internet** | Every `pip install` must happen in `setup.sh` on a login node. `setup.sh` sets `HTTP(S)_PROXY` to `proxy.alcf.anl.gov:3128` in case egress is not preset. |
| **`torch` from the conda module is matched to the driver** | The venv is created with `--system-site-packages` so it inherits that build. Installing your own `torch` is the standard way to get a CUDA/driver mismatch here. |
| **`/home` is small and slow** | If the repo is under `/home`, `setup.sh` says so. Prefer `/eagle/<project>/$USER` or `/grand/<project>/$USER`, and list the matching filesystem in `-l filesystems=`. |
| **`-l filesystems=` is mandatory** | Jobs are rejected without it. Both scripts request `home:eagle`; change it if your repo lives on `grand`. |
| **PBS starts you in `$HOME`** | Both scripts `cd "$PBS_O_WORKDIR"`. Always `qsub` from the repo root. |
| **Triton/Inductor caches** | Pinned per-job under the repo (`TRITON_CACHE_DIR`, `TORCHINDUCTOR_CACHE_DIR`) so concurrent jobs cannot corrupt each other's compilation cache. Delete `.triton-cache-*` when done. |
| **Module versions drift** | Both scripts use unversioned `module load conda`. If it fails, `module avail conda` and pin the version. |

## Queues

| queue | use | limits |
|---|---|---|
| `debug` | `job_smoke.pbs` | ≤ 1 h, ≤ 2 nodes, fast turnaround |
| `preemptable` | `job_experiments.pbs` (default) | starts sooner, can be killed mid-run |
| `prod` | guaranteed slot | longer queue wait |

The experiment job writes one file per experiment, so a preemption costs the tail
of the run rather than all of it. Switch `-q preemptable` to `-q prod` if you'd
rather wait than repeat.

## What comes back, and what it decides

Everything lands in `results/` stamped with the date and job id.

| file | decides |
|---|---|
| `gonogo-*.txt` | **the project.** Correctness gate, then dilated-8 + residue-perm-8 against our unpermuted kernel *and* FlexAttention at both its default 128×128 and its best over a block-size sweep. Permutation cost timed separately and never folded in. |
| `smoke-*.txt` | **gate.** Kernel vs dense reference across 7 tile shapes × 5 masks, the permuted path, mask-definition agreement, and `BlockIndex` vs `polyattn.cost`. |
| `exp-*-exp1_tile_shape.txt` | mechanism 2 — does wall-clock track elements as `BQ` falls, or flatten? (`docs/NOTES.md` §3a) |
| `exp-*-exp2_class_a.txt` | mechanism 1 — the go/no-go. dilated-8 + residue-perm-8, predicted 7.79× fewer elements. Permutation cost reported separately. (§4) |
| `exp-*-exp3_selection.txt` | **the top-ranked contribution.** Does the wall-clock argmin match the element-count argmin? Any disagreement kills novelty item 1 in its current form. (§5b, §5e) |
| `exp-*-exp4_flex_baseline.txt` | FlexAttention at BLOCK_SIZE 128/64/32 — the incumbent at a tile size it can actually use. (§3a-bis) |

## If the smoke test fails

Likely in this order, given the code has never run:

1. **Triton API drift** — `tl.dot(p, v, acc)` 3-arg accumulate and `tl.trans` have
   moved between Triton versions. Check `triton.__version__` in the env output.
2. **Dynamic loop bound** — the kernel loops `for j in range(0, n_blocks)` with
   `n_blocks` loaded from memory. If your Triton rejects it, hoist to a static
   `MAXKV` bound with an `if j < n_blocks` guard.
3. **Numerical tolerance** — `TOL = 3e-3` in `gpu/test_correctness.py` for fp16
   against an fp32 reference. Widen only after confirming the *pattern* of error
   is round-off and not a wrong tile set.
4. **Masks disagreeing** — if step 1 of the correctness suite fails, the numpy and
   torch definitions have diverged and every downstream number is meaningless.
   Fix that before anything else.

## Reporting back

The interesting result is a **gap between the predicted element count and the
measured time** — both columns are printed side by side for exactly that reason.
A clean agreement validates the cost model; a divergence is the more valuable
outcome and the one the analytical work cannot produce on its own.
