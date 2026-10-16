#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_instance_bench_20261016
PY="$ROOT/jev_vision_env/bin/python"
export NO_PROXY="${NO_PROXY:-},mirrors.aliyun.com,localhost,127.0.0.1"
export no_proxy="$NO_PROXY"
export PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple
"$PY" -m pip install --no-deps 'salesforce-lavis==1.0.2' 'timm==0.6.13' 'huggingface-hub==0.25.2'
"$PY" -m pip install 'numpy<2' 'fairscale==0.4.13' 'transformers==4.44.2' webdataset decord opendatasets pycocoevalcap ftfy sentencepiece einops spacy 'seaborn==0.13.2' google-crc32c
"$PY" -m pip install --no-deps "$ROOT/repos/MobileSAM"
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 "$PY" -c 'import torch; from lavis.models import load_model_and_preprocess; import mobile_sam; print("JEV_VISION_IMPORTS_READY")'
