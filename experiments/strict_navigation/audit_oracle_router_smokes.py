"""Learned oracle-router smoke audit; checks passthrough against frozen unmodified baseline."""
from pathlib import Path
import json,hashlib,time
import numpy as np
from audit_executed_trace import audit_executed_trace
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'oracle_waypoint_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
q=read(P/'protocol.json');done={x['tag'] for x in read(R/'status.json')['jobs'] if x['status']=='complete'}
for f,h in q['source_sha256'].items():assert sha(Path(f))==h
checks=[]
for j in q['smoke_jobs']:
 tag=j['tag'];v=j['model'];mode=j['mode'];assert tag in done
 p=R/'runs'/tag;meta=read(p/'case_metadata.json');metric=read(p/'metrics.json');rows=read(p/'cases_results.json');route=read(p/'oracle_waypoint_metadata.json')
 assert len(rows)==len(meta['cases'])==len(route['cases'])==2
 assert sha(Path(j['case_file']))==meta['case_sha256']==q['case_sha256']
 assert read(p/'model_metadata.json')['model_path']==q['models'][v]['weights']
 assert meta['normalization_sha256']==q['models'][v]['normalization_sha256']
 assert sha(Path(q['models'][v]['weights']))==q['model_sha256'][v]
 assert route['privileged'] and route['mode']==mode and not route['buffer_flushing']
 assert route['final_environment_goal_unchanged'] and metric['telemetry']['candidate_model_steps']>0
 a=read(p/'map_adapter_audit.json');assert all(x['start_render_max_error']==x['goal_render_max_error']==0 and x['start_state_error']<1e-5 for x in a)
 tr=np.load(p/'trace.npz');ta=audit_executed_trace(tr,rows)
 for rec,c in zip(route['cases'],meta['cases']):assert rec['route'][-1]==c['goals']['75']['xy']
 diff=None
 if mode=='final':
  base=R/'runs'/q['baseline_tags'][v];bm=read(base/'case_metadata.json');br=read(base/'cases_results.json');bt=np.load(base/'trace.npz')
  assert bm['cases']==meta['cases'] and bm['normalization_sha256']==meta['normalization_sha256']
  assert [x['success'] for x in br]==[x['success'] for x in rows]
  diff={}
  for k in tr.files:
   assert tr[k].shape==bt[k].shape
   assert np.allclose(tr[k],bt[k],rtol=1e-5,atol=1e-5,equal_nan=True),(tag,k)
   good=np.isfinite(tr[k])&np.isfinite(bt[k]);diff[k]=float(np.max(np.abs(tr[k][good].astype(float)-bt[k][good].astype(float)))) if good.any() else 0.
  assert metric['telemetry']['candidate_model_steps']==read(base/'metrics.json')['telemetry']['candidate_model_steps']
 checks.append({'tag':tag,'passed':True,'trace':ta,'passthrough_max_difference':diff,'route':route})
assert len(checks)==4
result={'passed':True,'updated_unix':time.time(),'checks':checks,'formal_evaluation_not_queued':True,'limits':'Learned router interface only, not evidence that waypoints improve navigation.'}
(P/'model_smoke_audit.json').write_text(json.dumps(result,indent=2));print('ORACLE_MODEL_SMOKES_PASSED',flush=True)
