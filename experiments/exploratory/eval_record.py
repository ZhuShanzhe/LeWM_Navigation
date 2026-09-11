"""Evaluation wrapper: official episode selection and solver, trace logging, explicit history."""
import os,sys,time,json,random,runpy,functools
from pathlib import Path
import numpy as np,torch
import stable_worldmodel as swm
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'round12h_20260909';repo=ROOT/'le-wm'
sys.path.insert(0,str(repo));os.chdir(repo)
tag=os.environ['LEWM_RUN_TAG'];out=R/'runs'/tag;out.mkdir(parents=True,exist_ok=True)
seed=next((int(a.split('=',1)[1]) for a in sys.argv if a.startswith('seed=')),42)
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(8)
def cv(x):
 if isinstance(x,dict):return {str(k):cv(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [cv(v) for v in x]
 if isinstance(x,(np.ndarray,torch.Tensor)):return x.detach().cpu().tolist() if isinstance(x,torch.Tensor) else x.tolist()
 if isinstance(x,np.generic):return x.item()
 return x
orig_load=swm.wm.utils.load_pretrained
def load(*a,**kw):
 model=orig_load(*a,**kw)
 hs=int(model.predictor.pos_embedding.shape[1])
 # Local jepa.py hardcodes rollout history=3 even when the model was trained with 1.
 if model.__class__.__module__=='jepa':
  model.rollout=functools.partial(model.rollout,history_size=hs)
 (out/'model_metadata.json').write_text(json.dumps({'class':str(type(model)),'history':hs,'parameters':sum(p.numel() for p in model.parameters()),'local_rollout_history_bound':model.__class__.__module__=='jepa'},indent=2))
 return model
swm.wm.utils.load_pretrained=load
orig_eval=swm.World.evaluate
def evaluate(self,*a,**kw):
 kw['video']=out/'video' if os.environ.get('LEWM_VIDEO','1')=='1' else None
 if isinstance(self.policy,swm.policy.RandomPolicy):self.policy.set_seed(seed)
 (out/'invocation.json').write_text(json.dumps({'argv':sys.argv,'seed':seed,'episodes':cv(kw.get('episodes_idx')),'start_steps':cv(kw.get('start_steps')),'goal_offset':kw.get('goal_offset'),'eval_budget':kw.get('eval_budget'),'rng_policy':'global RNGs and RandomPolicy action-space RNG seeded','history_note':'local JEPA uses checkpoint positional capacity'},indent=2))
 trace=[];original_step=self.envs.step
 def step(actions,*aa,**kk):
  result=original_step(actions,*aa,**kk)
  info=result[-1]
  record={'actions':np.asarray(actions).copy(),'terminated':np.asarray(result[2]).copy()}
  for k in ['proprio','state','qpos','qvel','distance_to_target','privileged_block_0_pos']:
   if k in info:record[k]=np.asarray(info[k]).copy()
  trace.append(record)
  return result
 self.envs.step=step
 start=time.time();result=orig_eval(self,*a,**kw)
 if trace:np.savez_compressed(out/'trace.npz',**{k:np.stack([t[k] for t in trace]) for k in trace[0] if all(k in t for t in trace)})
 payload={'metrics':cv(result),'elapsed_seconds':time.time()-start,'gpu':torch.cuda.get_device_name(),'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'argv':sys.argv}
 (out/'metrics.json').write_text(json.dumps(payload,indent=2));print('METRICS',json.dumps(payload),flush=True)
 return result
swm.World.evaluate=evaluate
runpy.run_path(str(repo/'eval.py'),run_name='__main__')
