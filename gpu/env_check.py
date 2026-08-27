"""Run this FIRST. Everything else assumes it passes.

Prints exactly what the box is, because several claims in docs/NOTES.md depend on
tile-shape and tensor-core details that vary by architecture.
"""
import importlib.util as iu
import sys


def main():
    ok = True
    print("=== python / packages ===")
    print(f"  python {sys.version.split()[0]}")
    for pkg in ("torch", "triton", "polyattn"):
        spec = iu.find_spec(pkg)
        print(f"  {pkg:<10} {'present' if spec else 'ABSENT'}")
        ok &= spec is not None
    if not ok:
        print("\nInstall: pip install -e '.[dev]' && pip install -r gpu/requirements.txt")
        return 1

    import torch
    print("\n=== device ===")
    if not torch.cuda.is_available():
        print("  torch.cuda.is_available() == False -- nothing here will run")
        return 1
    i = torch.cuda.current_device()
    p = torch.cuda.get_device_properties(i)
    cap = torch.cuda.get_device_capability(i)
    print(f"  {p.name}  sm_{cap[0]}{cap[1]}  {p.total_memory/2**30:.1f} GiB"
          f"  {p.multi_processor_count} SMs")
    print(f"  torch {torch.__version__}, cuda {torch.version.cuda}")
    import triton
    print(f"  triton {triton.__version__}")

    print("\n=== relevant to specific claims ===")
    print(f"  sm_{cap[0]}{cap[1]}: "
          + ("Hopper/Blackwell -- wgmma, m64nNk16 shapes" if cap[0] >= 9
             else "Ampere/Ada -- mma m16n8k16" if cap[0] >= 8
             else "pre-Ampere: fp16 tensor cores are m16n8k8; NOTES assumes >= sm_80"))
    print("  NOTES sec 7a treats the 16-wide tile floor as a stand-in, not a constant.")
    print("  This box's real floor is whatever exp1 measures, not what the ISA says.")

    try:
        from torch.nn.attention.flex_attention import create_block_mask  # noqa: F401
        print("  flex_attention: available (baseline for exp4)")
    except Exception as e:
        print(f"  flex_attention: UNAVAILABLE ({type(e).__name__}) -- exp4 will skip")

    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
