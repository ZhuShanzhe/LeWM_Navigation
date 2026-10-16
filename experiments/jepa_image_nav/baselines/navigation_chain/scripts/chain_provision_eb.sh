#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_instance_bench_20261016
if [[ ! -x "$ROOT/vln_env/bin/python" ]]; then
  /root/autodl-tmp/jepa_vln_baselines_20261015/policy_env/bin/python -m venv --system-site-packages "$ROOT/vln_env"
fi
unset https_proxy http_proxy HTTPS_PROXY HTTP_PROXY ALL_PROXY all_proxy
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
"$ROOT/vln_env/bin/python" -m pip install 'transformers==4.51.3' accelerate 'ai2thor==5.0.0' \
  'numpy==1.26.4' 'opencv-python-headless==4.11.0.86' 'opencv-python==4.11.0.86' \
  'scipy==1.15.3' openai omegaconf hydra-core safetensors \
  qwen-vl-utils pillow imageio tqdm flask requests
"$ROOT/vln_env/bin/python" -m pip freeze > "$ROOT/results/vln_environment.txt"
