# Sourceable. Repairs the environment the conda modulefile fails to finish
# building. Safe to source more than once; prints only when it does something.
#
# WHY THIS EXISTS
# The ALCF docs (Polaris > Data Science > Frameworks > PyTorch) say conda's
# torch is compiled from source, finds CUDA via CUDA_HOME, and has an `mpi`
# backend "built against the system's HPE Cray MPICH libraries". So torch has a
# hard link-time dependency on cray-mpich. `module load conda` normally pulls
# cray-mpich in as part of its dependency chain -- but on this system that chain
# aborts on an unknown cray-hdf5-parallel version, so cray-mpich is never
# loaded and torch dies with:
#     libmpi_gnu_123.so.12: cannot open shared object file
# The 123 encodes GCC 12.3. The library exists on disk; only the module that
# would have put it on LD_LIBRARY_PATH failed. So find it and add it.

# CUDA_HOME, per the docs' `echo $CUDA_HOME` example.
if [ -z "${CUDA_HOME:-}" ]; then
  for c in /soft/compilers/cudatoolkit/cuda-12.4.1 /soft/compilers/cudatoolkit/cuda-*; do
    [ -d "$c" ] && { export CUDA_HOME="$c"; echo "  env_fixup: CUDA_HOME=$CUDA_HOME"; break; }
  done
fi

_polyattn_torch_err() { python -c 'import torch' 2>&1 | tail -1; }

if ! python -c 'import torch' >/dev/null 2>&1; then
  _err="$(_polyattn_torch_err)"
  case "$_err" in
    *libmpi*)
      # e.g. "libmpi_gnu_123.so.12: cannot open shared object file"
      _lib="$(printf '%s' "$_err" | grep -o 'libmpi[a-z0-9_]*\.so[0-9.]*' | head -1)"
      echo "  env_fixup: torch needs $_lib, which the aborted conda modulefile never provided"
      _dir=""
      for root in /opt/cray/pe/mpich /opt/cray/pe/lib /opt/cray; do
        [ -d "$root" ] || continue
        _dir="$(find "$root" -name "$_lib" -printf '%h\n' 2>/dev/null | head -1)"
        [ -n "$_dir" ] && break
      done
      if [ -n "$_dir" ]; then
        export LD_LIBRARY_PATH="$_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        echo "  env_fixup: found it in $_dir -- prepended to LD_LIBRARY_PATH"
        if python -c 'import torch' >/dev/null 2>&1; then
          echo "  env_fixup: torch imports now"
        else
          echo "  env_fixup: still failing: $(_polyattn_torch_err)"
        fi
      else
        echo "  env_fixup: $_lib not found under /opt/cray."
        echo "             Fall back to a self-contained wheel:"
        echo "               pip install --ignore-installed torch --index-url https://download.pytorch.org/whl/cu124"
      fi
      ;;
    *libcuda*)
      echo "  env_fixup: libcuda missing -- expected on a LOGIN node, fatal on a compute node."
      ;;
  esac
  unset _err _lib _dir
fi
unset -f _polyattn_torch_err

# --- Triton's host compiler ---------------------------------------------
# Triton JIT-compiles a small C helper (backends/nvidia/driver.c) at first use
# and picks the compiler from $CC. On Polaris the Cray/NVHPC programming
# environment sets CC=nvc, but Triton passes GCC-specific flags, so nvc dies:
#     nvc-Error-Unknown switch: -Wno-psabi
# It is not a CUDA problem and not a Triton bug -- just the wrong host compiler.
# Point CC at a real gcc. Triton only needs it to build that one shim.
case "$(basename "${CC:-}" 2>/dev/null)" in
  nvc|nvc++|cc|CC|craycc)  _polyattn_badcc=1 ;;
  "")                      _polyattn_badcc=1 ;;   # unset: Triton may still find nvc
  *)                       _polyattn_badcc=0 ;;
esac
if [ "${_polyattn_badcc:-0}" = "1" ]; then
  _polyattn_oldcc="${CC:-unset}"
  for _g in /usr/bin/gcc "$(command -v gcc 2>/dev/null)" /usr/bin/cc; do
    if [ -x "${_g:-}" ] && "$_g" --version 2>/dev/null | head -1 | grep -qi gcc; then
      export CC="$_g"
      export CXX="${_g}++"; [ -x "$CXX" ] || export CXX="$(command -v g++ 2>/dev/null || echo "$_g")"
      echo "  env_fixup: CC=$CC  (was '$_polyattn_oldcc'; Triton needs gcc, not nvc)"
      break
    fi
  done
fi
unset _polyattn_badcc _g _polyattn_oldcc
