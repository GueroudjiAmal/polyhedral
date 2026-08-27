#!/usr/bin/env bash
# Run in order. Stop at the first failure -- later numbers are meaningless
# if the kernel is wrong.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}

echo "############ env ############";        $PY gpu/env_check.py
echo "############ correctness ####";        $PY gpu/test_correctness.py
echo "############ exp1 tile shape #";       $PY gpu/exp1_tile_shape.py
echo "############ exp2 class A ####";       $PY gpu/exp2_class_a.py
echo "############ exp3 selection ##";       $PY gpu/exp3_selection.py
echo "############ exp4 flex base ##";       $PY gpu/exp4_flex_baseline.py
