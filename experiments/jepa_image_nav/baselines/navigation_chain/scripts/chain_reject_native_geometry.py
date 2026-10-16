"""Explicitly exclude the full-native-geometry run from performance comparisons."""
import gzip,json
from pathlib import Path
import numpy as np
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
run=ROOT/'results/goat_native_hssd300'
violations=[];episodes=0
for path in run.glob('*/*.steps.jsonl.gz'):
    episodes+=1
    with gzip.open(path,'rt') as stream:
        steps=[json.loads(line) for line in stream]
        for s,n in zip(steps,steps[1:]):
            if s['action']!=1:
                distance=float(np.linalg.norm(np.array(n['position'])-s['position']))
                if distance>1e-5:violations.append(distance)
records=[json.loads(p.read_text()) for p in run.glob('*/*.json')]
records=[r for r in records if 'episode_id' in r]
assert len(records)==300 and violations
result={'acceptance':'rejected','performance_score_valid':False,'episode_records':len(records),
    'episodes_with_trace':episodes,'record_status_counts':dict(__import__('collections').Counter(r['status'] for r in records)),
    'non_translation_actions_with_position_change':len(violations),
    'maximum_unintended_displacement_m':max(violations),
    'reason':'Frozen common-geometry starts were reused with a changed native body/navmesh; turn/STOP position invariance failed',
    'permitted_use':'Engineering diagnosis only; not an official or cross-domain performance score',
    'controlled_alternative':'goat_native_camera_common_geometry300'}
(run/'rejected_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
