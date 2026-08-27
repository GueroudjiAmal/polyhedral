#!/usr/bin/env bash
# Start the chain. Submits ONE job; each job launches its successor when it
# finishes, so only one is ever in the queue -- the debug queue rejects a batch
# submission with "would exceed per-user limit of jobs in Q state", and a
# dependency chain does not help because dependent jobs still occupy Q.
#
#   ./polaris/submit_all.sh <project>          # whole chain from the gate
#   ./polaris/submit_all.sh <project> fixed    # resume the chain at `fixed`
#
# Stop it:  touch polaris/STOP      Resume:  rm polaris/STOP && ./submit_all.sh ...
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
A="${1:?usage: submit_all.sh <project> [start-at-job]}"
START="${2:-gate}"

grep -qx "$START" polaris/jobs/ORDER || {
  echo "unknown job '$START'. Order is:"; sed 's/^/  /' polaris/jobs/ORDER; exit 1; }

printf 'ACCT=%s
' "$A" > polaris/.chain.conf
rm -f polaris/STOP

id=$(qsub -A "$A" "polaris/jobs/$START.pbs")
echo "chain started at '$START':  $id"
echo
echo "remaining, in order:"
awk -v s="$START" 'f||$0==s{f=1; print "  " $0}' polaris/jobs/ORDER
echo
echo "watch:    qstat -u \$USER"
echo "results:  results/<job>/latest/out.txt"
echo "halt:     touch polaris/STOP   (current job finishes, successor not submitted)"
