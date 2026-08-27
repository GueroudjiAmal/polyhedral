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
# EXACTLY the sequence in the ALCF docs (Polaris > Data Science > Python and
# > PyTorch). Nothing invented:
#     module use /soft/modulefiles
#     module load conda
#     conda activate base
#
# Two things were previously here and are gone on purpose:
#   * `module reset` -- not in the docs, and on a Cray PE it can drop the very
#     modules (cray-mpich, cray-hdf5) the conda build links against.
#   * a `frameworks` fallback -- that is the Aurora module name. Polaris uses
#     `conda`. Trying it here only produced a misleading second failure.
module use /soft/modulefiles

if ! module load conda 2>/tmp/.modload.$$; then
  cat /tmp/.modload.$$ >&2
  cat >&2 <<'HINT'

FATAL: `module load conda` failed.

If it reported a dependency as UNKNOWN (cray-hdf5-parallel, gcc-native, ...)
that dependency almost certainly EXISTS -- Lmod is reading a stale user cache
written before a site software update. Lmod says so itself in the error text.
Clear it and start a fresh shell:

    rm -rf ~/.lmod.d/.cache ~/.cache/lmod
    module --ignore_cache spider conda        # confirm which versions are real
    module use /soft/modulefiles && module load conda

If you have a saved module collection it can pin retired modules too:

    module savelist        # if a "default" exists and you did not want it:
    module disable default

Only if a specific conda version is genuinely broken site-side, pick another:

    module --ignore_cache avail conda

HINT
  rm -f /tmp/.modload.$$
  exit 1
fi
rm -f /tmp/.modload.$$

set +u                       # conda's activate scripts trip `set -u`
conda activate base
set -u
python -c 'import sys; print("   python", sys.version.split()[0], sys.executable)'
LOADED_MODULE=conda

# TWO WAYS TO GET TORCH. Default is the module's build; --pip-torch is the
# escape hatch when the module's build is unusable.
#
#   inherit (default)  --system-site-packages, uses the conda module's torch.
#                      Matched to the driver, no download. But on Polaris that
#                      torch is an HPC build linked against Cray MPICH, and it
#                      fails with e.g. "libmpi_gnu_123.so.12: cannot open shared
#                      object file" whenever the PrgEnv it was built against has
#                      been retired -- which is a site upgrade away at any time.
#
#   --pip-torch        clean venv, torch from PyPI. NO Cray MPI dependency at
#                      all, which is the right trade here: every experiment is
#                      single-GPU (select=1, CUDA_VISIBLE_DEVICES=0) and nothing
#                      in gpu/ calls MPI. Costs a ~2.5GB download through the
#                      ALCF proxy; must match the compute-node driver.
PIP_TORCH=0
for a in "$@"; do [ "$a" = "--pip-torch" ] && PIP_TORCH=1; done

if [ "$PIP_TORCH" = "1" ]; then
  echo "== building a CLEAN venv with torch from PyPI (no Cray MPI)"
  [ -d .venv-polaris ] || python -m venv .venv-polaris
  source .venv-polaris/bin/activate
  python -m pip install --upgrade pip
  python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
else
  echo "== building a venv that INHERITS the module's torch"
  echo "   (re-run with --pip-torch if 'import torch' fails on a Cray library)"
  [ -d .venv-polaris ] || python -m venv --system-site-packages .venv-polaris
  source .venv-polaris/bin/activate
  python -m pip install --upgrade pip
fi

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
        print("     -> NOT a driver issue.")
        if "libmpi" in msg or "cray" in msg.lower():
            print("        This is CRAY MPI, not CUDA. The conda module's torch is an HPC")
            print("        build linked against cray-mpich for a specific GCC (the number")
            print("        in libmpi_gnu_NNN encodes it). When the site retires that")
            print("        PrgEnv, this torch can no longer load and no module fixes it.")
            print("        Nothing in gpu/ uses MPI -- every run is single-GPU -- so the")
            print("        right answer is to stop inheriting it:")
            print("            rm -rf .venv-polaris && bash polaris/setup.sh --pip-torch")
        else:
            print("        Check:  module list")
            print("                echo $LD_LIBRARY_PATH | tr : '\\n' | grep -iE 'cuda|nvidia'")
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
