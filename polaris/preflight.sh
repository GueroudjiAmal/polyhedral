#!/usr/bin/env bash
# Run on a Polaris LOGIN node BEFORE qsub. Every check here corresponds to a
# real failure mode that PBS reports only as H (held) or an instant E (exiting)
# with an empty or missing log -- states that tell you nothing about the cause.
#
#   usage:  bash polaris/preflight.sh <your_project>

proj="${1:-}"
fail=0
note() { printf '  %-8s %s\n' "$1" "$2"; }
ok()   { note OK "$1"; }
bad()  { note FAIL "$1"; fail=1; }
warn() { note warn "$1"; }

cd "$(dirname "${BASH_SOURCE[0]}")/.."
echo "== preflight in $PWD"

# 1. -o logs/ is resolved by PBS BEFORE the script body runs. The scripts do
#    `mkdir -p logs` internally, which is too late: if logs/ is absent at submit
#    time PBS cannot stage the output file and the job goes straight to E.
[ -d logs ] && ok "logs/ exists (PBS -o target)" \
             || bad "logs/ missing -- PBS -o logs/ cannot stage output. mkdir -p logs"

# 2. compute nodes have no outbound network, so the venv must already exist.
if [ -f polaris/env.generated.sh ]; then
  . polaris/env.generated.sh
  [ -f "$POLYATTN_VENV/bin/activate" ] && ok "venv present: $POLYATTN_VENV" \
    || bad "venv $POLYATTN_VENV missing -- re-run polaris/setup.sh"
  python -c 'import torch' 2>/dev/null && ok "torch imports on login node" \
    || warn "torch does not import here; fine if it is only libcuda, fatal if Cray MPI"
else
  bad "polaris/env.generated.sh missing -- run polaris/setup.sh on a login node"
fi

# 3. an invalid or placeholder project is accepted by qsub, then HELD forever.
if [ -z "$proj" ]; then
  bad "no project given -- usage: bash polaris/preflight.sh <your_project>"
else
  if command -v sbank >/dev/null 2>&1 && sbank-list-allocations 2>/dev/null | grep -q "$proj"; then
    ok "project $proj has an allocation"
  else
    warn "cannot verify project '$proj' from here; confirm with: sbank-list-allocations"
  fi
fi
grep -q '^#PBS -A' polaris/job_*.pbs 2>/dev/null \
  && bad "a job script still hard-codes #PBS -A; pass -A at submit time instead" \
  || ok "no hard-coded #PBS -A"

# 4. requesting a filesystem the project cannot reach is also a hold.
for fsdir in /eagle /grand; do
  [ -d "$fsdir" ] || warn "$fsdir not visible from this node"
done
grep -h '^#PBS -l filesystems=' polaris/job_gonogo.pbs | sed 's/^/  decl:    /'
warn "if your allocation is on grand not eagle, edit filesystems= or the job is HELD"

# 5. running from /home is slow and quota-limited, and Triton caches are large.
case "$PWD" in /home/*) warn "running from /home; prefer /eagle/<project>/$USER" ;; esac

# 6. debug queue limits: <=2 nodes, <=1h.
awk '/^#PBS -l walltime=/{print "  decl:    " $0}' polaris/job_gonogo.pbs

echo
[ "$fail" -eq 0 ] && echo "preflight clean -- qsub -A $proj polaris/job_gonogo.pbs" \
                  || echo "preflight FAILED -- fix the above before qsub"
exit "$fail"

# Control-flow dry run. Two exp8 jobs died on the GPU with an IndexError that
# py_compile could not see; this stubs torch/triton and runs every experiment's
# main() end to end on CPU in seconds. It found three real bugs the first time.
if python tools/dryrun_experiments.py >/tmp/.pa_dry 2>&1; then
  ok "all experiments' control flow runs (tools/dryrun_experiments.py)"
else
  bad "an experiment would die on the GPU -- see below"
  tail -6 /tmp/.pa_dry | sed 's/^/         /'
fi
