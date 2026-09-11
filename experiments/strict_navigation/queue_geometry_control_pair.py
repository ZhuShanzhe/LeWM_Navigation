"""Finite matched single/multi-layout control, gated on successful interface audits."""
import json,copy,hashlib,time
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent;PY=str(ROOT/'venv/bin/python')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert json.loads((R/'collections/audit.json').read_text())['passed']
assert json.loads((R/'collections/model_smoke_audit.json').read_text())['passed']
p=R/'jobs.json';jobs=json.loads(p.read_text());known={j['tag'] for j in jobs};new=[]
for variant in ['single','multi']:
 smoke='geometry_smoke_'+variant+'_s4001'+('_v2' if variant=='single' else '')
 tag='geometry_control_'+variant+'_s4001'
 tr=copy.deepcopy(next(j for j in jobs if j['tag']==smoke))
 tr.update(tag=tag,training=True,watchdog_seconds=43200)
 tr['argv']=[a.replace(smoke,tag) for a in tr['argv']]
 tr['env'].pop('LEWM_STOP_STEPS',None);tr['env']['LEWM_TRAIN_SECONDS']='43000'
 tr['note']='Fresh 10-epoch paired control. Same seed, architecture, objective and update budget. Not smoke continuation.'
 new.append(tr)
 for split,axis,door in [('train',1,49),('validation',1,65),('validation',1,129),('validation',0,65),('validation',0,129)]:
  for b in range(2):
   src=f'strict3072_map_{split}_a{axis}_d{door}_b{b}'
   etag=f'geometry_{variant}4001_map_{split}_a{axis}_d{door}_b{b}'
   j=copy.deepcopy(next(x for x in jobs if x['tag']==src));j['tag']=etag
   j['argv']=[a.replace('strict_tw_s3072_v2',tag).replace(src,etag) for a in j['argv']]
   j['env']['LEWM_NORMALIZATION_FILE']=str(R/'collections'/variant/'normalization.json')
   j['note']='Common development cases; fixed original CEM cost and cap. Original geometry control plus 4 validation geometries. Not untouched final test.'
   new.append(j)
new.append({'tag':'audit_geometry_control_pair4001','watchdog_seconds':600,'argv':[PY,str(R/'audit_geometry_control_pair.py')],'env':{'CUDA_VISIBLE_DEVICES':''}})
new=[j for j in new if j['tag'] not in known]
idx=next(i for i,j in enumerate(jobs) if j['tag']=='strict3073_validation_long_base_b1')+1
jobs[idx:idx]=new
assert len({j['tag'] for j in jobs})==len(jobs)
protocol={'created_unix':time.time(),'seed':4001,'fresh_training':True,'epochs':10,'expected_steps':51920,'train_frames':736684,'train_clips':664684,'validation_clips':83354,'eval_cases_per_map':100,'maps':['train_axis1_door49','validation_axis1_door65','validation_axis1_door129','validation_axis0_door65','validation_axis0_door129'],'selection':'last full 10-epoch weights, not best navigation validation','cap':{'execution_steps':150,'horizon_model_steps':5,'receding_horizon_model_steps':5,'cem_samples':300,'cem_iterations':10,'topk':30,'warm_start':False},'limits':['one paired training seed; not cross-seed robustness','same 100 coordinate cases reused across maps; do not pool as independent trials','original map is training geometry but states are independently fixed development cases','geometry variants remain two rooms/one door, not new topology','both datasets contain matched geometry-guided collection prior','test geometry and final reserve not used for this stage','single and multi use their own training-only normalization; not a pure one-factor coordinate transformation'],'files':{v:{'manifest_sha256':sha(R/'collections'/v/'manifest.json'),'normalization_sha256':sha(R/'collections'/v/'normalization.json')} for v in ['single','multi']}}
q=R/'geometry_control_protocol.json'
if not q.exists():q.write_text(json.dumps(protocol,indent=2))
tmp=R/'jobs.control_pair.tmp';tmp.write_text(json.dumps(jobs,indent=2));tmp.replace(p)
print('QUEUED',len(new),'TOTAL',len(jobs),'after strict3073 validation')
