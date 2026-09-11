"""Record official evaluation results; keep the model and planning algorithm unchanged."""
import json,os,runpy,sys,time,random
from pathlib import Path
import numpy as np
import torch
import stable_worldmodel as swm
root=Path('/root/autodl-tmp/lewm_research'); repo=root/'le-wm'
sys.path.insert(0,str(repo)); os.chdir(repo)
seed=next((int(a.split('=',1)[1]) for a in sys.argv[1:] if a.startswith('seed=')),42)
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
torch.set_num_threads(8)
tag=os.environ.get('LEWM_RUN_TAG','eval');folder=root/'runs'/tag;folder.mkdir(parents=True,exist_ok=True)
original=swm.World.evaluate
def serializable(v):
 if isinstance(v,dict): return {str(k):serializable(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)): return [serializable(x) for x in v]
 if isinstance(v,np.ndarray): return v.tolist()
 if isinstance(v,np.generic): return v.item()
 if isinstance(v,torch.Tensor): return v.detach().cpu().tolist()
 return v
def evaluate(self,*args,**kwargs):
 kwargs['video']=folder
 is_random=isinstance(self.policy,swm.policy.RandomPolicy)
 if is_random: self.policy.set_seed(seed)
 (folder/'invocation.json').write_text(json.dumps({'argv':sys.argv,'seed':seed,
 'random_action_seed':seed if is_random else None,'wrapper_version':2,
 'episodes':serializable(kwargs.get('episodes_idx')),'start_steps':serializable(kwargs.get('start_steps'))},indent=2))
 start=time.time();result=original(self,*args,**kwargs)
 payload={'metrics':serializable(result),'elapsed_seconds':time.time()-start,'seed':seed,
 'random_action_seed':seed if is_random else None,'wrapper_version':2,
 'argv':sys.argv,'gpu':torch.cuda.get_device_name(),'max_cuda_allocated_bytes':torch.cuda.max_memory_allocated()}
 (folder/'metrics.json').write_text(json.dumps(payload,indent=2))
 print('RECORDED_METRICS',json.dumps(payload),flush=True)
 return result
swm.World.evaluate=evaluate
runpy.run_path(str(repo/'eval.py'),run_name='__main__')
