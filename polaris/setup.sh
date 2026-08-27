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

echo "== repo: $REPO"
case "$REPO" in
  /home/*) echo "   NOTE: /home on Polaris is small and not the fast filesystem."
           echo "   Consider /eagle/<project>/$USER or /grand/<project>/$USER instead." ;;
esac

module use /soft/modulefiles
module load conda            # check 'module avail conda' if this fails
conda activate base

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

echo
echo "== setup done. Next:"
echo "   edit polaris/job_*.pbs and set -A <your_project>"
echo "   qsub polaris/job_smoke.pbs"
