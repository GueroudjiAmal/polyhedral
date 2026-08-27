"""Execute a notebook in place, storing its outputs. Usage: execute_nb.py [path]"""
import pathlib
import sys

import nbformat
from nbclient import NotebookClient

ROOT = pathlib.Path(__file__).resolve().parents[1]
path = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "notebooks" / "01_reasoning_log.ipynb"

nb = nbformat.read(path, as_version=4)
NotebookClient(nb, timeout=2400, kernel_name="python3",
               resources={"metadata": {"path": str(path.parent)}}).execute()
nbformat.write(nb, path)

errs = [o for c in nb.cells for o in c.get("outputs", []) if o.output_type == "error"]
figs = [o for c in nb.cells for o in c.get("outputs", []) if "image/png" in o.get("data", {})]
print(f"{path.name}: {len(nb.cells)} cells, {len(figs)} figures, {len(errs)} errors")
for e in errs:
    print("\n".join(e.get("traceback", []))[-1500:])
sys.exit(1 if errs else 0)
