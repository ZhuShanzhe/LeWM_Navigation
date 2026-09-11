#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/lewm_research/env.sh
root=/root/autodl-tmp/lewm_research
for seed in 42 43 44; do
 tag=random_seeded_default_s$seed
 echo "START $tag $(date -Iseconds)"
 LEWM_RUN_TAG="$tag" python "$root/run_eval.py" --config-name tworoom policy=random seed=$seed "output.filename=$root/runs/${tag}_results.txt" > "$root/logs/${tag}.log" 2>&1
 echo "DONE $tag $(date -Iseconds)"
done
echo ALL_DONE
