#!/usr/bin/env bash
set -euo pipefail
source /root/autodl-tmp/lewm_research/env.sh
root=/root/autodl-tmp/lewm_research
for i in $(seq 1 240); do
  if [ -f "$root/runs/train_pilot500/training_summary.json" ]; then break; fi
  if ! kill -0 11652 2>/dev/null; then echo "Training stopped without summary"; exit 1; fi
  sleep 5
done
test -f "$root/runs/train_pilot500/training_summary.json"
run_eval () {
 tag="$1"; shift
 echo "START $tag $(date -Iseconds)"
 LEWM_RUN_TAG="$tag" python "$root/run_eval.py" --config-name tworoom "$@" "output.filename=$root/runs/${tag}_results.txt" > "$root/logs/${tag}.log" 2>&1
 echo "DONE $tag $(date -Iseconds)"
}
for seed in 43 44; do
 run_eval pretrained_default_s$seed policy="$root/data/hf_tworoom" seed=$seed
 run_eval random_default_s$seed policy=random seed=$seed
done
run_eval pretrained_long75_s42 policy="$root/data/hf_tworoom" seed=42 eval.goal_offset_steps=75 eval.eval_budget=150
run_eval random_long75_s42 policy=random seed=42 eval.goal_offset_steps=75 eval.eval_budget=150
run_eval pilot500_default_s42 policy="$root/data/checkpoints/train_pilot500/weights_epoch_5.pt" seed=42
echo "ALL_DONE $(date -Iseconds)"
