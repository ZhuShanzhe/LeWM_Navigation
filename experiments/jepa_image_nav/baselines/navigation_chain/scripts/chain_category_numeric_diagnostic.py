"""Compare stored SPL and float64-reconstructed paths without changing scores."""
import gzip,json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
run=R/'results/goat_category_hssd300_v3'
differences=[]
for path in run.glob('*/*.steps.jsonl.gz'):
    record=json.loads(path.with_name(path.name.replace('.steps.jsonl.gz','.json')).read_text())
    with gzip.open(path,'rt') as stream:
        steps=[json.loads(line) for line in stream]
    length64=sum(float(np.linalg.norm(np.array(s['after'])-s['before'])) for s in steps)
    length32=sum(float(np.linalg.norm(np.array(s['after'],dtype=np.float32)-np.array(s['before'],dtype=np.float32))) for s in steps)
    d=record['initial_verified_view_geodesic_m']
    expected64=record['verified_view_success']*d/max(d,length64,1e-8)
    expected32=record['verified_view_success']*d/max(d,length32,1e-8)
    differences.append(dict(episode=record['episode_id'],path_error64=length64-record['path_length_m'],
        path_error32=length32-record['path_length_m'],spl_error64=expected64-record['verified_view_spl'],
        spl_error32=expected32-record['verified_view_spl'],length=record['path_length_m']))
result={'records':len(differences),'max_abs':{key:max(abs(x[key]) for x in differences) for key in
    ['path_error64','path_error32','spl_error64','spl_error32']},
    'largest_spl_errors':sorted(differences,key=lambda x:abs(x['spl_error64']),reverse=True)[:5]}
(run/'numerical_diagnostic.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
