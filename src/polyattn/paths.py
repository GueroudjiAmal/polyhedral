"""Repository-relative paths, so scripts and notebooks agree regardless of cwd."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
NOTEBOOKS = ROOT / "notebooks"

for _p in (RESULTS, FIGURES):
    _p.mkdir(parents=True, exist_ok=True)
