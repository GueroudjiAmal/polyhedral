# Running the GPU harness on Polaris (ALCF)

Polaris: 4× **A100 40GB** per node (sm_80), AMD EPYC Milan host, **PBS Pro**
scheduler. Nothing here needs more than one GPU on one node — these are
single-kernel microbenchmarks, not a scaling study.

> **The kernel has never been executed anywhere.** It was written on a machine
> with no CUDA device. `job_smoke.pbs` is a gate, not a formality — expect to fix
> something on the first submission.

## Three commands

**Run everything as small jobs (recommended):**

```bash
./polaris/setup.sh                                  # LOGIN node, once
./polaris/preflight.sh <your_project>
./polaris/submit_all.sh <your_project>              # gate + 10 jobs
```

Eleven small jobs instead of one three-hour block: ten fit the **debug** queue
(≤50 min) and schedule far sooner, and a failure or preemption costs one
experiment rather than the run. The gate goes first; everything else is submitted
with `-W depend=afterok:<gate>` so nothing burns a slot if the kernel is broken.

**Each job writes its own directory**, with a `latest` symlink:

```
results/<job>/<stamp>/meta.txt   job, host, GPU, CC, exit code
results/<job>/<stamp>/env.txt    env_check output
results/<job>/<stamp>/out.txt    the experiment
results/<job>/latest -> <stamp>
```

Read them with `cat results/*/latest/out.txt`, or one at a time.

Subset: `./polaris/submit_all.sh <project> cell3 fixed` — still gated.

| job | queue | wall | decides |
|---|---|---|---|
| `gate` | debug | 20m | **nothing else counts if this fails** |
| `cell3` | debug | 20m | the only parameter-free falsifier in the set |
| `gonogo` | debug | 30m | headline, fresh-compiled FlexAttention baseline |
| `cells` | debug | 40m | all three cells, paired/interleaved |
| `fixed` | debug | 40m | out-of-sample test of the fixed-cost account |
| `imbal` | debug | 30m | counting vs makespan, wave-count control |
| `select` | debug | 50m | wall-clock argmin vs element argmin |
| `amort` | debug | 40m | permutation once-per-forward (**prefill only**) |
| `tiles` | debug | 50m | `p(BQ,A,mask)` — read across masks |
| `classa` | debug | 30m | class A across tile shapes |
| `traffic` | preemptable | 90m | lower bound on a class that is modelled, not implemented |

**Caveat on concurrency:** ALCF limits how many jobs one user can have running in
`debug`. If that limit is 1, these serialise rather than fanning out — still
better than one long block, because results arrive incrementally and the queue
wait per job is short, but do not expect all ten at once. If they serialise and
you want them faster, move the later ones to `preemptable` in
`tools/gen_polaris_jobs.py` and regenerate.

The jobs are **generated** — edit the table in `tools/gen_polaris_jobs.py` and
rerun it, never the `.pbs` files. Ten hand-maintained job scripts drift, and that
has already cost this project two queue slots.

---

```bash
# 1. on a LOGIN node (compute nodes have no outbound network)
./polaris/setup.sh

# 2. check everything the queue will check, before the queue does
./polaris/preflight.sh

# 3. gate first, then the experiments. PASS -A ON THE COMMAND LINE.
qsub -A <your_project> polaris/job_gonogo.pbs      # ~10 min, debug queue
qstat -u $USER
#   ... read results/gonogo-*.txt in full ...
qsub -A <your_project> polaris/job_experiments.pbs # only if the go/no-go warrants it
```

The scripts deliberately carry **no `#PBS -A` directive**. PBS Pro accepts an
invalid account at submit time and then holds the job indefinitely with no log
written — a placeholder left in the file fails *silently*. Passing `-A` on the
command line fails *loudly* if you forget it.

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
| `exp-*-exp8_three_cells.txt` | **the three cells that measure a coefficient rather than a rate.** Two bracket the contiguity coefficient from opposite sides; the third tests the wave account with no free parameter. N is load-bearing in the third — at a convenient N it silently returns a null. |
| `exp-*-exp4_flex_baseline.txt` | FlexAttention at BLOCK_SIZE 128/64/32 — the incumbent at a tile size it can actually use. (§3a-bis) |

## If the job goes to H or E instead of running

**Get the actual reason first** — everything below is a guess until you do:

```bash
qstat -sf <jobid> | grep -i comment      # PBS states the hold reason verbatim
```

Three causes were found in these scripts and fixed; if you hit a fourth, that
command will name it.

| state | cause | status |
|---|---|---|
| **H** (held) | invalid or missing `-A` account. PBS accepts it at submit, then holds forever with no log. | fixed — `#PBS -A` removed, pass `-A` on the command line |
| **E** (exits at once) | `#PBS -o logs/` with no `logs/` directory. PBS resolves the output path *before* the script body runs, so a `mkdir` inside the job is too late. | fixed — `logs/` is in the repo and `setup.sh` creates it |
| **E** (exits at once) | `set -euo pipefail` + `source .venv-polaris/bin/activate` when the venv does not exist, i.e. `setup.sh` was never run. Instant non-zero exit before anything runs. Separately, `set -u` breaks `conda activate`, whose scripts reference unset variables. | fixed — activation wrapped in `set +u`, explicit FATAL if the venv is missing |
| **H** | `-l filesystems=home:eagle` when your allocation is on **grand**. Requesting a filesystem you cannot access is itself a hold. | **RESOLVED** — the repo lives under `/lus/eagle/projects/radix-io/...`, so `home:eagle` is correct as written. |
| **E** on the login node, before any job | The site's conda modulefiles have a broken dependency chain: `conda/2025-09-25` (the DEFAULT) wants `gcc-native/14.2` and `cray-hdf5-parallel/1.14.3.5`; `conda/2024-04-29` wants `cray-hdf5-parallel/1.12.2.9`. All reported UNKNOWN, cache cleared, no change. | **not ours to fix** — belongs in a ticket to support@alcf.anl.gov. Everything here works around it. |
| **E** in the job, after the module fix | `module load` was treated as FATAL. It only sets environment variables; whether the job can do anything is decided by the venv's interpreter. A cosmetic site failure was killing work that would otherwise run. | fixed — the module load is best-effort and warns; the job gates on `import torch` **and** `torch.cuda.is_available()` instead |

`polaris/preflight.sh` checks all of these on a login node before you submit.

## The module name is discovered, not assumed

`setup.sh` tries a fallback list and writes the one that worked to
`polaris/env.generated.sh`. The three job scripts **source that file** rather than
hard-coding a name, and fail loudly if it is missing.

This matters because the site's DEFAULT conda is broken -- see the table below.
It is the *version* that has to be pinned, not the name. Three `.pbs` files each assuming
`conda` would have failed in the queue exactly the way `setup.sh` failed on the
login node — with a queue wait in between.

If `setup.sh` cannot load anything, it prints `module avail` and stops. The two
diagnostics worth running by hand at that point:

```bash
module use /soft/modulefiles && module avail conda   # versions, not just the default
module spider gcc-native            # is the dependency it wants actually installed?
```

## Known good configuration

Observed from an actual run, so these are facts rather than defaults:

- repo at `/lus/eagle/projects/radix-io/<user>/...` → `-l filesystems=home:eagle`
  is correct, and eagle is the right place for it (not `/home`)
- project appears to be `radix-io` — still pass it explicitly as
  `qsub -A radix-io ...` rather than hard-coding a `#PBS -A` back in

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
