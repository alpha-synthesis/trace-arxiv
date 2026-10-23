#!/usr/bin/env bash
# R1: the price-informed labels (r0) + corrected code, both sign variants.
# Needs results/r0 (data bundle) and prices (tools/download_prices.py).
set -uo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
mkdir -p logs
for v in datasign llmsign; do
  echo "=== $(date -Is) R1 $v start"
  $PY scripts/run_pipeline.py configs/r1_$v.yaml > logs/r1_$v.log 2>&1
  rc=$?
  echo "=== $(date -Is) R1 $v exit=$rc"
  [ $rc -eq 0 ] || exit $rc
done
