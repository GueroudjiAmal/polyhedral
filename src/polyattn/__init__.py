"""Polyhedral analysis of sparse-attention masks.

Three experiments, all exact element counts, none yet measured on hardware:

  experiments.granularity  how much work a boolean block mask wastes
  experiments.reindex      which legal changes of basis remove that waste
  experiments.compose      splitting a mask so each part gets its own basis

docs/NOTES.md carries the reasoning log; notebooks/ carries the executable version.
"""
from . import cost, masks, paths, shapes, transforms  # noqa: F401

__all__ = ["masks", "cost", "transforms", "shapes", "paths"]
