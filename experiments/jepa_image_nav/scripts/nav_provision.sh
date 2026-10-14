#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_vln_data_20261014
mkdir -p "$ROOT/tools" "$ROOT/manifests" "$ROOT/source" "$ROOT/raw" "$ROOT/datasets" "$ROOT/reports" "$ROOT/logs"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export CONDA_REMOTE_READ_TIMEOUT_SECS=120 CONDA_REMOTE_CONNECT_TIMEOUT_SECS=30
/root/miniconda3/bin/conda create --prefix "$ROOT/env" --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge -c https://conda.anaconda.org/aihabitat -y python=3.9 habitat-sim=0.3.3 headless withbullet pip numpy pillow scipy
"$ROOT/env/bin/python" -m pip install --index-url https://pypi.tuna.tsinghua.edu.cn/simple 'huggingface_hub<1' hf_xet tqdm pyyaml
"$ROOT/env/bin/python" -c 'import habitat_sim;print("HABITAT_SIM",habitat_sim.__version__)'
echo ENV_READY
