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

# --- conda base + venv, exactly per the ALCF docs -----------------------
#   https://docs.alcf.anl.gov/polaris/data-science/python/
#     module use /soft/modulefiles; module load conda; conda activate base
#     CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
#     VENV_DIR="$(pwd)/venvs/${CONDA_NAME}"
#     python -m venv "${VENV_DIR}" --system-site-packages
#     source "${VENV_DIR}/bin/activate"
#
# One deviation, and it is also from the docs: base is read-only, so a broken
# base package is replaced by SHADOWING it --
#     python3 -m pip install --ignore-installed <package>
# That is how we deal with base's torch when it cannot load (see below).

module use /soft/modulefiles

if [ -n "${CONDA_PREFIX:-}" ] && python -c 'import sys' 2>/dev/null; then
  # base is already active in this shell -- reuse it rather than re-running
  # `module load conda`, which resolves to the site DEFAULT version and can
  # fail on a dependency the system does not have.
  echo "== conda base already active: $CONDA_PREFIX"
else
  module load conda || {
    echo >&2 "FATAL: module load conda failed. If it names an unknown dependency"
    echo >&2 "       version, that is a site issue -- report to support@alcf.anl.gov."
    echo >&2 "       Meanwhile: module --ignore_cache avail conda   and load one that works."
    exit 1
  }
  set +u; conda activate base; set -u
fi

CONDA_NAME=$(echo ${CONDA_PREFIX} | tr '\/' '\t' | sed -E 's/mconda3|\/base//g' | awk '{print $NF}')
VENV_DIR="${REPO}/venvs/${CONDA_NAME}"
echo "== venv: $VENV_DIR"
mkdir -p "$VENV_DIR"
[ -f "$VENV_DIR/bin/activate" ] || python -m venv "$VENV_DIR" --system-site-packages
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

# Does the INHERITED torch actually load? base's is an HPC build linked against
# cray-mpich for a specific GCC; when the site retires that PrgEnv it raises
# e.g. "libmpi_gnu_123.so.12: cannot open shared object file". Nothing in gpu/
# uses MPI, so shadow it with a PyPI wheel via the documented --ignore-installed.
# Try the surgical repair BEFORE downloading 2.5GB: the conda modulefile aborts
# before it puts cray-mpich on LD_LIBRARY_PATH, and that library is on disk.
set +u; . "${REPO}/polaris/env_fixup.sh"; set -u

echo "== checking inherited torch"
if python -c 'import torch' 2>/tmp/.torchchk.$$; then
  python -c 'import torch; print("   inherited torch OK", torch.__version__, torch.version.cuda)'
  TORCH_SOURCE=module      # needs the conda module's environment at run time
else
  echo "   inherited torch FAILED:"; sed 's/^/     /' /tmp/.torchchk.$$ | tail -3
  echo "   -> shadowing it with a PyPI wheel (docs: pip install --ignore-installed)"
  python -m pip install --ignore-installed torch \
      --index-url https://download.pytorch.org/whl/cu124
  TORCH_SOURCE=pypi        # self-contained: bundles its own CUDA, needs no module
fi
rm -f /tmp/.torchchk.$$

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
echo "   venv: $VENV_DIR"

# Hand the discovered module name to the job scripts. They must not hard-code it:
# three .pbs files silently assuming a module name (or the site DEFAULT version)
# would fail in the queue the same way this script can fail on the login node.
# NOTE: do not add a `frameworks` fallback -- that is the Aurora module name, it
# is not documented for Polaris, and trying it only produces a second misleading
# failure on top of the real one.
cat > polaris/env.generated.sh <<EOSTAMP
# GENERATED by polaris/setup.sh on $(date -Is) -- do not edit, re-run setup.sh.
POLYATTN_MODULE="conda/${CONDA_NAME}"
POLYATTN_VENV="$VENV_DIR"
POLYATTN_TORCH_SOURCE="${TORCH_SOURCE}"
EOSTAMP
echo "   wrote polaris/env.generated.sh"
echo "   bash polaris/preflight.sh <your_project>"
echo "   qsub -A <your_project> polaris/job_gonogo.pbs"
