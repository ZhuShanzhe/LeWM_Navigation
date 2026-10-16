#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_instance_bench_20261016
export NO_PROXY=localhost,127.0.0.1
export no_proxy="$NO_PROXY"
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=1
# Completed episode files are resumed without re-scoring or replacement.
# Process boundaries release GPU memory/lock between frozen task prefixes.
for limit in 10 20 30 40 50 60; do
  "$ROOT/vln_env/bin/python" -u "$ROOT/tools/chain_eb_navigation.py" --limit "$limit"
done
"$ROOT/goat_env/bin/python" -u "$ROOT/tools/chain_eb_acceptance.py"
