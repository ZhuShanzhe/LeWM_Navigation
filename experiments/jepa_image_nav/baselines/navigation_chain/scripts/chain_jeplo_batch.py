"""Loopback-only native JEPLO deployment repeats, separate from navigation SR."""
from collections import defaultdict
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
NATIVE = Path('/root/autodl-tmp/jeplo_eval_20260928/official_alignment')
PYTHON = NATIVE / 'runtime_env/bin/python'
OUT = ROOT / 'results/jeplo_native_motion_validated'
OUT.mkdir(parents=True, exist_ok=True)

with (ROOT / 'gpu_benchmark.lock').open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    plan = {'profile': 'jeplo-official-readme-sim-20260928',
        'source_commit':'460e2272907af053d6e66b57fec71fdbe6ca1774',
        'case_durations_seconds':{'stand':12,'forward':12,'rotate':12,'stairs':45,'boxes':35},
        'repeats_per_case':3, 'simulation_only':True, 'network':'loopback, DDS domain 1',
        'paper_success_rate_reproduction':False,
        'criteria':'Upstream simulation/controller/sensors and weights; frozen prior self-defined deployment commands/ascent gates',
        'randomness':'Upstream cage random_device remains enabled; repeats are not claimed as controlled training seeds'}
    (OUT / 'plan.json').write_text(json.dumps(plan, indent=2))
    records = []
    for case, duration in plan['case_durations_seconds'].items():
        for repeat in range(3):
            run = OUT / (case+'_'+str(repeat+1))
            summary = run / 'summary.json'
            if summary.exists():
                value = json.loads(summary.read_text())
            else:
                command = [str(PYTHON),str(ROOT/'tools/jeplo_native_smoke.py'),'--root',str(NATIVE),
                    '--output',str(run),'--duration',str(duration),'--case',case]
                if case in {'stairs','boxes'}:
                    command += ['--stop-at-ascent-goal']
                with (OUT/(case+'_'+str(repeat+1)+'.log')).open('w') as output:
                    proc = subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=180)
                if not summary.exists():
                    value = {'complete':False,'runtime_failure':True,'return_code':proc.returncode}
                else:
                    value = json.loads(summary.read_text())
            value['case'], value['repeat'] = case, repeat+1
            records.append(value)
            (OUT/'records.json').write_text(json.dumps(records,indent=2))
            print(json.dumps({'case':case,'repeat':repeat+1,'complete':value.get('complete'),
                              'ascent_goal_held':value.get('ascent_goal_held')}),flush=True)
    report = dict(plan, completed_runs=len(records),
        successful_runtime_runs=sum(bool(r.get('complete')) for r in records), records=records)
    (OUT/'summary.json').write_text(json.dumps(report,indent=2))
