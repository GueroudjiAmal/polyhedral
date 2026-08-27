#!/usr/bin/env bash
# ONE-TIME SETUP -- run on a Polaris LOGIN node, not in a job.
#
# Compute nodes have no outbound network, so every download has to happen here.
# Login nodes have no GPU, so nothing is verified here either -- that is what
# job_smoke.pbs is for.
set -euo pipefail

# ALCF egress proxy. Login nodes usually have it preset; harmless if redundant.
export HTTP_PROXY=${HTTP_PROXY:-http://proxy.alcf.anl.gov:3128}
export HTTPS_PROXY=${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}
export http_proxy=$HTTP_PROXY  https_proxy=$HTTPS_PROXY

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# PBS resolves the -o path BEFORE the job body runs, so a mkdir inside the job
# is too late and the job exits (E) with no log written. Create it here.
mkdir -p logs results

echo "== repo: $REPO"
case "$REPO" in
  /home/*) echo "   NOTE: /home on Polaris is small and not the fast filesystem."
           echo "   Consider /eagle/<project>/$USER or /grand/<project>/$USER instead." ;;
esac

# --- module stack -------------------------------------------------------
# `module load conda` can fail when the login shell already has a PrgEnv/compiler
# combination that the conda modulefile cannot swap to -- it reports the missing
# dependency (e.g. gcc-native, cray-hdf5-parallel) rather than the real cause,
# which is a dirty or stale module environment. So: reset first, ignore the
# cache, and fall back across the plausible module names rather than assuming.
module use /soft/modulefiles 2>/dev/null || true

echo "== resetting module environment"
module reset            2>/dev/null || module purge 2>/dev/null || true
module use /soft/modulefiles 2>/dev/null || true

load_python_stack() {
  for m in "$@"; do
    echo "-- trying: module load $m"
    if module --ignore_cache load "$m" 2>/dev/null && command -v python >/dev/null; then
      echo "   loaded $m"; LOADED_MODULE="$m"; return 0
    fi
    module unload "$m" 2>/dev/null || true
  done
  return 1
}

# `frameworks` is the newer ALCF name for the ML stack; `conda` the older one.
if ! load_python_stack frameworks conda; then
  echo
  echo "FATAL: could not load a Python/ML module. What is actually available:"
  module --ignore_cache avail frameworks conda 2>&1 | sed 's/^/    /'
  echo
  echo "Then re-run with the working name, e.g.:"
  echo "    module use /soft/modulefiles && module load <name>/<version>"
  echo "If a dependency is reported missing, check it exists:  module spider gcc-native"
  exit 1
fi

set +u                                  # conda activate trips `set -u`
conda activate base 2>/dev/null || echo "   (no conda base to activate; continuing)"
set -u
python -c 'import sys; print("   python", sys.version.split()[0], sys.executable)'

# --system-site-packages inherits the module's torch build, which is matched to
# the driver. Installing our own torch is the usual way to get a CUDA mismatch.
if [ ! -d .venv-polaris ]; then
  python -m venv --system-site-packages .venv-polaris
fi
source .venv-polaris/bin/activate

python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

python - <<'PY'
import importlib.util as u
have = {p: u.find_spec(p) is not None for p in ("torch", "triton")}
print("  torch:", have["torch"], " triton:", have["triton"])
if not all(have.values()):
    print("  -> installing missing pieces; if this pulls a new torch, check it")
    print("     matches the driver with: python -c 'import torch;print(torch.version.cuda)'")
PY
python -m pip install -r gpu/requirements.txt || {
  echo "pip install failed -- if torch already came from the conda module, that is fine."; }

# --- torch import diagnostic --------------------------------------------
# A bare `import torch` traceback here is misleading: on a LOGIN node the most
# common cause is simply that libcuda.so.1 does not exist (no GPU, no driver),
# which is EXPECTED and not a setup failure. A different missing library is a
# real problem. Distinguish them instead of printing a traceback.
echo
echo "== torch import check (login node -- no GPU here by design)"
python - <<'PYCHK'
import ctypes, traceback
try:
    ctypes.CDLL("libcuda.so.1")
    print("  libcuda.so.1: present")
except OSError as e:
    print(f"  libcuda.so.1: ABSENT ({e})")
    print("     ^ normal on a Polaris login node. Only a problem if it also")
    print("       happens inside a job -- that is what job_smoke.pbs checks.")
try:
    import torch
    print(f"  import torch: OK  {torch.__version__}  cuda {torch.version.cuda}")
except Exception as e:
    msg = str(e)
    print(f"  import torch: FAILED  {type(e).__name__}: {msg}")
    if "libcuda" in msg:
        print("     -> driver library only. Expected on a login node; verify in a job.")
    else:
        print("     -> NOT a driver issue. Likely the module environment is missing a")
        print("        runtime dependency this torch build needs. Check:")
        print("          module list")
        print("          echo $LD_LIBRARY_PATH | tr : '\n' | grep -iE 'cuda|nvidia|nvhpc'")
        print("        and try loading conda WITHOUT the `module reset` above.")
PYCHK

echo
echo "== setup done. Next:"
echo "   module stack that worked: ${LOADED_MODULE:-unknown}"

# Hand the discovered module name to the job scripts. They must not hard-code it:
# if the site's `conda` modulefile is broken and `frameworks` is the one that
# works, three .pbs files silently assuming `conda` would fail in the queue the
# same way this script just failed on the login node.
cat > polaris/env.generated.sh <<EOSTAMP
# GENERATED by polaris/setup.sh on $(date -Is) -- do not edit, re-run setup.sh.
POLYATTN_MODULE="${LOADED_MODULE:-conda}"
POLYATTN_VENV="$REPO/.venv-polaris"
EOSTAMP
echo "   wrote polaris/env.generated.sh (module=${LOADED_MODULE:-conda})"
echo "   put that same name in polaris/job_*.pbs if it is not 'conda'"
echo "   bash polaris/preflight.sh <your_project>"
echo "   qsub -A <your_project> polaris/job_gonogo.pbs"
