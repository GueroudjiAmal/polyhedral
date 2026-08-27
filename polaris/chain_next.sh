#!/usr/bin/env bash
# Launch the successor of $1. Called by a running job as its last act.
#   $1 = name of the job that just finished    $2 = its exit code
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
me="${1:?}"; rc="${2:-0}"

[ -f polaris/STOP ] && { echo "chain: STOP present, not continuing"; exit 0; }
[ -f polaris/.chain.conf ] || { echo "chain: no .chain.conf, not continuing"; exit 0; }
. polaris/.chain.conf                       # provides ACCT

# The gate is the only job whose failure stops the chain. A single broken
# experiment should not cost the other nine their slots.
if [ "$me" = gate ] && [ "$rc" -ne 0 ]; then
  echo "chain: GATE FAILED (rc=$rc) -- stopping. Nothing downstream would mean anything."
  exit 0
fi

next=$(awk -v me="$me" 'f{print; exit} $0==me{f=1}' polaris/jobs/ORDER)
if [ -z "$next" ]; then echo "chain: $me was last -- done."; exit 0; fi

id=$(qsub -A "$ACCT" "polaris/jobs/$next.pbs" 2>&1) &&   echo "chain: $me -> $next  $id" ||   echo "chain: FAILED to submit $next: $id" >&2
