"""CPU audit of controlled-data model interface checks; not an accuracy evaluation."""
import json,hashlib
from pathlib import Path
import numpy as np,torch
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
assert json.loads((R/'collections/audit.json').read_text())['passed']
result={}
for variant in ['single','multi']:
 tr=R/'runs'/(f'geometry_smoke_{variant}_s4001'+('_v2' if variant=='single' else ''));ev=R/'runs'/f'geometry_smoke_{variant}_eval'
 summary=json.loads((tr/'training_summary.json').read_text())
 assert summary['global_step']==20 and not summary['full_epoch_budget_completed']
 manifest=R/'collections'/variant/'manifest.json';norm=manifest.parent/'normalization.json'
 split=json.loads((tr/'split_audit.json').read_text())
 assert split['passed'] and split['manifest_sha256']==hashlib.sha256(manifest.read_bytes()).hexdigest()
 assert split['episode_counts']=={'train':8000,'validation':1000,'test':0}
 assert split['clip_counts']['train']==664684 and split['clip_counts']['validation']==83354
 meta=json.loads((ev/'case_metadata.json').read_text())
 assert meta['normalization_sha256']==hashlib.sha256(norm.read_bytes()).hexdigest()
 assert meta['normalization_path']==str(norm) and len(meta['cases'])==2
 rows=json.loads((ev/'cases_results.json').read_text());assert len(rows)==2
 assert all(np.isfinite(x['endpoint_distance']) for x in rows)
 map_audit=json.loads((ev/'map_adapter_audit.json').read_text());assert len(map_audit)==2
 assert all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in map_audit)
 metrics=json.loads((ev/'metrics.json').read_text());assert metrics['telemetry']['candidate_model_steps']>0
 trace=np.load(ev/'trace.npz');assert all(np.isfinite(trace[k]).all() for k in trace.files)
 weights=Path(summary['weights']);assert weights.is_file() and weights.stat().st_size>1000000
 result[variant]={'passed':True,'steps':20,'split_audit':split,'normalization_sha256':meta['normalization_sha256'],'model_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),'cases':2,'checkpoint':str(weights)}
assert result['single']['split_audit']['clip_counts']==result['multi']['split_audit']['clip_counts']
payload={'passed':True,'full_training_requires_stage_review':True,'variants':result}
(R/'collections/model_smoke_audit.json').write_text(json.dumps(payload,indent=2))
print('CONTROLLED_MODEL_INTERFACE_PASSED',list(result))
