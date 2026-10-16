#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_instance_bench_20261016
PY="$ROOT/policy_env/bin/python"
source /etc/network_turbo >/dev/null 2>&1 || true
export NO_PROXY="${NO_PROXY:-},localhost,127.0.0.1,mirrors.aliyun.com"
export no_proxy="$NO_PROXY"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=12.0
export MAX_JOBS=4
"$PY" -m pip install 'transformers==4.44.2' 'scipy==1.15.3' addict yapf termcolor
"$PY" -u "$ROOT/tools/chain_setup_unigoal.py"
"$PY" -m pip install --no-deps --no-build-isolation "$ROOT/repos/segment-anything"
"$PY" -m pip install --no-deps --no-build-isolation /root/autodl-tmp/jepa_vln_baselines_20261015/repos/LightGlue
"$PY" -m pip install --no-deps --no-build-isolation "$ROOT/repos/GroundingDINO"
cd "$ROOT/repos/UniGoal"
"$PY" -c 'from groundingdino import _C; from src.graph.graph import Graph; from src.agent.unigoal.agent import UniGoal_Agent; from src.map.bev_mapping import BEV_Map; print("UNIGOAL_FULL_IMPORTS_READY")'
"$PY" -m pip freeze > "$ROOT/results/unigoal_environment.txt"
