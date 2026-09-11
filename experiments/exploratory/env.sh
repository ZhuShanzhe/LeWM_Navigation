#!/usr/bin/env bash
source /root/autodl-tmp/lewm_research/env.sh
export LD_LIBRARY_PATH=/root/autodl-tmp/lewm_research/round12h_20260909/runtime/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}
export PYOPENGL_PLATFORM=egl
export MUJOCO_GL=egl
export SPT_CACHE_DIR=/root/autodl-tmp/lewm_research/round12h_20260909/spt_cache
