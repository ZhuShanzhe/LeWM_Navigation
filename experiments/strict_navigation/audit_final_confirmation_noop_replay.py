"""Audit-only amendment: replay zero-action boundary projection; frozen experiments unchanged."""
from pathlib import Path
import json,hashlib,os
import numpy as np,torch
from stable_worldmodel.envs.two_room.env import TwoRoomEnv
torch.set_num_threads(1)
ROOT=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
noop_diagnostics={}
def verify_noop(tag,trace,rows,cases,case_data):
 actions=trace['actions']
 assert np.isfinite(actions).all() and np.all(actions==0),tag
 pos=trace['proprio'].reshape(len(trace['proprio']),50,-1)[:,:,:2]
 env=TwoRoomEnv(init_value=case_data.get('init_value'));env.reset(seed=9106201)
 moved=[]
 for i,(row,case) in enumerate(zip(rows,cases)):
  env._set_state(np.asarray(case['start_xy'],np.float32))
  env._set_goal_state(np.asarray(case['goals']['75']['xy'],np.float32))
  _,_,term,_,_=env.step(np.zeros(2,np.float32));first=env.agent_position.numpy().copy()
  assert np.allclose(pos[:row['steps'],i],first,rtol=0,atol=1e-5),(tag,i,'zero-action-replay')
  assert bool(row['success'])==bool(term),(tag,i,'zero-action-success')
  env.step(np.zeros(2,np.float32))
  assert np.allclose(env.agent_position.numpy(),first,rtol=0,atol=1e-5),(tag,i,'not-stationary-after-projection')
  displacement=float(np.linalg.norm(first-np.asarray(case['start_xy'])))
  assert np.isclose(displacement,row['path_length'],rtol=0,atol=1e-5),(tag,i,'path')
  if displacement>1e-5:moved.append({'episode':case['episode'],'start_step':case['start_step'],'initial_xy':case['start_xy'],'projected_xy':first.tolist(),'displacement':displacement,'success':row['success']})
 env.close()
 noop_diagnostics[tag]={'passed':True,'n':len(rows),'initial_projection_cases':moved}
amendment=json.loads((ROOT/'final_confirmation_v1/audit_amendment_noop.json').read_text())
assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==amendment['wrapper_sha256']
source=(ROOT/'audit_final_confirmation.py').read_text()
needle="  if method=='noop':assert all(x['path_length']<1e-5 for x in rows)"
assert source.count(needle)==1
source=source.replace(needle,"  if method=='noop':verify_noop(tag,tr,rows,meta['cases'],expected)")
source=source.replace("'requires_final_research_synthesis':True}","'requires_final_research_synthesis':True,'audit_amendment':amendment,'noop_replay_audits':noop_diagnostics}")
namespace={'verify_noop':verify_noop,'noop_diagnostics':noop_diagnostics,'amendment':amendment}
if os.environ.get('LEWM_AUDIT_PREFIX_ONLY')=='1':
 source=source.split('groups={}')[0].replace("for job in q['evaluations']:","for job in [j for j in q['evaluations'] if j['tag'] in done]:")
exec(compile(source,str(ROOT/'audit_final_confirmation.py')+'[noop-replay-amendment]','exec'),namespace)
if os.environ.get('LEWM_AUDIT_PREFIX_ONLY')=='1':
 result={'passed':True,'completed_batches_audited':len(namespace['records']),'noop_replay_audits':noop_diagnostics,'amendment':amendment}
 (ROOT/'reports/final_prefix_noop_replay_audit.json').write_text(json.dumps(result,indent=2))
 print('PREFIX_REPLAY_AUDIT_PASSED',len(namespace['records']),json.dumps(noop_diagnostics),flush=True)
