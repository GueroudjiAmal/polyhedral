#!/usr/bin/env bash
# Bundle whatever has finished so far into one pasteable file.
#   ./tools/collect.sh              everything that exists
#   ./tools/collect.sh cell3 gonogo just these
#
# meta.txt is included with every result on purpose: host, GPU and CC decide
# whether two numbers are comparable. Cross-job variance has been 8-23% on
# identical configurations, and one whole job was invalidated by the host
# compiler being nvc instead of gcc. A number without its meta is not readable.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
want=("$@")
for d in results/*/latest; do
  [ -e "$d" ] || continue
  n=$(basename "$(dirname "$d")")
  if [ ${#want[@]} -gt 0 ]; then printf '%s\n' "${want[@]}" | grep -qx "$n" || continue; fi
  echo "================================================== $n"
  sed 's/^/  /' "$d/meta.txt" 2>/dev/null
  echo "  ----------------------------------------------"
  cat "$d/out.txt" 2>/dev/null || echo "  (no out.txt -- job died before the experiment ran)"
  echo
done
# a job that dies in the preamble never writes results/, only a PBS log
shopt -s nullglob
for f in logs/pa-*.o*; do
  d="results/$(basename "$f" | sed 's/^pa-//;s/\.o[0-9]*$//')/latest"
  [ -e "$d/out.txt" ] && continue
  echo "================================================== PBS log (no results dir): $f"
  tail -40 "$f"; echo
done
