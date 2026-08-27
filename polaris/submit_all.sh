#!/usr/bin/env bash
# Submit every job. The gate goes first; everything else waits for it to PASS and
# then runs CONCURRENTLY -- ten small debug-queue jobs schedule far sooner than
# one three-hour block, and a failure or preemption costs one experiment.
#
#   ./polaris/submit_all.sh <project>            # gate + all, dependent
#   ./polaris/submit_all.sh <project> cell3 fixed  # just these, still after gate
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="${1:?usage: submit_all.sh <project> [job ...]}"; shift || true

GATE=$(qsub -A "$A" polaris/jobs/gate.pbs)
echo "gate           $GATE"

want=("$@")
for f in polaris/jobs/*.pbs; do
  n=$(basename "$f" .pbs); [ "$n" = gate ] && continue
  if [ ${#want[@]} -gt 0 ]; then
    printf '%s\n' "${want[@]}" | grep -qx "$n" || continue
  fi
  id=$(qsub -A "$A" -W depend=afterok:"$GATE" "$f")
  printf '%-14s %s\n' "$n" "$id"
done
echo
echo "watch:   qstat -u \$USER"
echo "results: results/<job>/latest/out.txt"
