"""Strict navigation evaluation: frozen cases/scaling; measured compute, no protocol mixing."""
import os,sys,json,time,hashlib,functools,runpy,random
from pathlib import Path
import numpy as np,torch
import stable_worldmodel as swm
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'strict_nav_20260910';OLD=ROOT/'round12h_20260909';repo=ROOT/'le-wm'
sys.path.insert(0,str(repo))
out=R/'runs'/os.environ['LEWM_RUN_TAG'];out.mkdir(parents=True,exist_ok=True)
stats_path=Path(os.environ.get('LEWM_NORMALIZATION_FILE',str(R/'splits/normalization.json')))
stats=json.loads(stats_path.read_text())
case_path=Path(os.environ['LEWM_CASE_FILE']);case_data=json.loads(case_path.read_text())
cases=case_data['cases'][:int(os.environ.get('LEWM_CASE_LIMIT','50'))]
telemetry={'cost_calls':0,'candidate_sequences':0,'candidate_model_steps':0,'encoder_images':0,'predictor_token_inputs':0,'solver_calls':[],'policy_calls_seconds':[]}
def synchronize():
 if torch.cuda.is_available():torch.cuda.synchronize()
orig_load=swm.wm.utils.load_pretrained
def load(*a,**kw):
 model=orig_load(*a,**kw)
 hs=int(model.predictor.pos_embedding.shape[1])
 if model.__class__.__module__=='jepa':model.rollout=functools.partial(model.rollout,history_size=hs)
 head_file=os.environ.get('LEWM_TEMPORAL_HEAD')
 if head_file:
  from reachability_model import TemporalHead
  ck=torch.load(head_file,map_location='cpu',weights_only=False)
  assert ck['checkpoint_sha256']==hashlib.sha256(Path(a[0]).read_bytes()).hexdigest(),'Head/model checkpoint mismatch'
  head=TemporalHead(ck['dim']).to(os.environ.get('LEWM_EVAL_DEVICE','cuda')).eval().requires_grad_(False)
  head.load_state_dict(ck['state_dict'])
  raw_criterion=model.criterion;mode=os.environ.get('LEWM_HEAD_COST','replace')
  assert mode in ['replace','hybrid']
  def criterion(info):
   pred=info['predicted_emb'][:,:,-1,:];goal=info['goal_emb']
   goal=goal[:,-1,:][:,None,:].expand_as(pred) if goal.ndim==3 else goal[:,:,-1,:].expand_as(pred)
   learned=head(pred,goal);telemetry['metric_head_pairs']=telemetry.get('metric_head_pairs',0)+pred.shape[0]*pred.shape[1]
   if mode=='replace':return learned
   raw=raw_criterion(info)
   def standard(x):return (x-x.mean(1,keepdim=True))/(x.std(1,keepdim=True,unbiased=False)+1e-6)
   return standard(raw)+standard(learned)
  model.criterion=criterion
  (out/'metric_head_metadata.json').write_text(json.dumps({'path':head_file,'sha256':hashlib.sha256(Path(head_file).read_bytes()).hexdigest(),'kind':ck['kind'],'cost':mode,'parameters':sum(p.numel() for p in head.parameters()),'model_sha256':ck['checkpoint_sha256'],'adapted':True,'extra_supervision':'training trajectory temporal separations only'},indent=2))
 original_cost=model.get_cost
 def cost(info,actions):
  if os.environ.get('LEWM_BOUND_ACTIONS')=='1':
   ss=stats['action'];m=torch.as_tensor(ss['mean'],device=actions.device,dtype=actions.dtype);sd=torch.as_tensor(ss['std'],device=actions.device,dtype=actions.dtype)
   assert actions.shape[-1]%len(m)==0
   reps=actions.shape[-1]//len(m);lo=((-1-m)/sd).repeat(reps);hi=((1-m)/sd).repeat(reps)
   actions.clamp_(min=lo,max=hi)
  telemetry['cost_calls']+=1;telemetry['candidate_sequences']+=int(actions.shape[0]*actions.shape[1])
  telemetry['candidate_model_steps']+=int(np.prod(actions.shape[:3]))
  return original_cost(info,actions)
 model.get_cost=cost
 def enc_hook(module,args):telemetry.__setitem__('encoder_images',telemetry['encoder_images']+int(args[0].shape[0]))
 def pred_hook(module,args):telemetry.__setitem__('predictor_token_inputs',telemetry['predictor_token_inputs']+int(np.prod(args[0].shape[:2])))
 model.encoder.register_forward_pre_hook(enc_hook);model.predictor.register_forward_pre_hook(pred_hook)
 (out/'model_metadata.json').write_text(json.dumps({'model_path':str(a[0]),'class':str(type(model)),'history':hs,'parameters':sum(p.numel() for p in model.parameters())},indent=2))
 return model
swm.wm.utils.load_pretrained=load
orig_solve=swm.solver.CEMSolver.solve
def solve(self,info,*a,**kw):
 synchronize();t=time.perf_counter();before=telemetry['candidate_model_steps']
 result=orig_solve(self,info,*a,**kw);synchronize()
 telemetry['solver_calls'].append({'seconds':time.perf_counter()-t,'n_active_envs':len(next(iter(info.values()))),'candidate_model_steps':telemetry['candidate_model_steps']-before})
 return result
swm.solver.CEMSolver.solve=solve
orig_action=swm.policy.WorldModelPolicy.get_action
def action(self,*a,**kw):
 synchronize();t=time.perf_counter();result=orig_action(self,*a,**kw);synchronize()
 telemetry['policy_calls_seconds'].append(time.perf_counter()-t);return result
swm.policy.WorldModelPolicy.get_action=action
if os.environ.get('LEWM_NOOP')=='1':
 swm.policy.RandomPolicy.get_action=lambda self,obs,**kw:np.zeros(self.env.action_space.shape,dtype=self.env.action_space.dtype)
orig_eval=swm.World.evaluate
def evaluate(self,*a,**kw):
 kw['episodes_idx']=[c['episode'] for c in cases];kw['start_steps']=[c['start_step'] for c in cases]
 kw['video']=out/'video' if os.environ.get('LEWM_VIDEO','0')=='1' else None
 if isinstance(self.policy,swm.policy.RandomPolicy):self.policy.set_seed(seed)
 meta={**case_data,'cases':cases,'case_sha256':hashlib.sha256(case_path.read_bytes()).hexdigest(),'goal_offset':kw['goal_offset'],'budget':kw['eval_budget'],'normalization':'strict training set only','normalization_path':str(stats_path),'normalization_sha256':hashlib.sha256(stats_path.read_bytes()).hexdigest(),'action_constraint':'project physical [-1,1] before scoring and CEM moment updates' if os.environ.get('LEWM_BOUND_ACTIONS')=='1' else 'original unconstrained search; environment clips physical actions','argv':sys.argv}
 (out/'case_metadata.json').write_text(json.dumps(meta,indent=2))
 if isinstance(self.policy,swm.policy.WorldModelPolicy):
  cfg=self.policy.cfg;s=self.policy.solver
  per_cap=int(np.ceil(kw['eval_budget']/(cfg.receding_horizon*cfg.action_block)))*s.n_steps*s.num_samples*cfg.horizon
  meta['compute_cap_per_case']={'candidate_model_steps':per_cap,'horizon':cfg.horizon,'iterations':s.n_steps,'candidates':s.num_samples,'receding':cfg.receding_horizon,'warm_start':cfg.warm_start}
  (out/'case_metadata.json').write_text(json.dumps(meta,indent=2))
 traces=[];step=self.envs.step
 def recording(actions,*aa,**kk):
  result=step(actions,*aa,**kk);info=result[-1]
  traces.append({'actions':np.asarray(actions).copy(),'terminated':np.asarray(result[2]).copy(),'proprio':np.asarray(info['proprio']).copy()})
  return result
 self.envs.step=recording
 synchronize();t=time.perf_counter();result=orig_eval(self,*a,**kw);synchronize();elapsed=time.perf_counter()-t
 if traces:np.savez_compressed(out/'trace.npz',**{k:np.stack([row[k] for row in traces]) for k in traces[0]})
 h=str(kw['goal_offset']);positions=np.stack([x['proprio'] for x in traces]);term=np.stack([x['terminated'] for x in traces])
 rows=[]
 for i,c in enumerate(cases):
  done=np.flatnonzero(term[:,i]);n=int(done[0]+1) if len(done) else len(traces)
  xy=positions[:n,i].reshape(n,-1)[:,:2];start=np.array(c['start_xy']);goal=np.array(c['goals'][h]['xy'])
  path=np.vstack([start,xy]);movement=np.linalg.norm(np.diff(path,axis=0),axis=1)
  rows.append({'episode':c['episode'],'start_step':c['start_step'],'success':bool(len(done)),'initial_success':c['goals'][h]['initially_within_success'],'steps':n,'endpoint_distance':float(np.linalg.norm(xy[-1]-goal)),'path_length':float(movement.sum()),'stalled_steps_lt_0_5px':int((movement<.5).sum()),'across_wall':c['goals'][h]['across_wall'],'initial_euclidean':c['goals'][h]['euclidean']})
 (out/'cases_results.json').write_text(json.dumps(rows,indent=2))
 def cv(x):
  if isinstance(x,dict):return {str(k):cv(v) for k,v in x.items()}
  if isinstance(x,(list,tuple)):return [cv(v) for v in x]
  if isinstance(x,np.ndarray):return x.tolist()
  if isinstance(x,np.generic):return x.item()
  return x
 payload={'metrics':cv(result),'elapsed_seconds':elapsed,'n_cases':len(cases),'telemetry':telemetry,'argv':sys.argv,'timing_note':'batched evaluation; single-env runs separately identify online latency; video disabled by default'}
 (out/'metrics.json').write_text(json.dumps(payload,indent=2))
 print('STRICT_METRICS',json.dumps({'n':len(rows),'success_rate':100*np.mean([x['success'] for x in rows]),'candidate_model_steps':telemetry['candidate_model_steps'],'elapsed_seconds':elapsed}),flush=True)
 return result
swm.World.evaluate=evaluate
seed=next((int(a.split('=',1)[1]) for a in sys.argv if a.startswith('seed=')),42)
random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);torch.set_num_threads(8)
source=(repo/'eval.py').read_text()
source=source.replace('model.to("cuda")', 'model.to(os.environ.get("LEWM_EVAL_DEVICE", "cuda"))')
source=source.replace('config_path="./config/eval"','config_path="'+str(repo/'config/eval')+'"')
old='        processor.fit(col_data)'
new="""        ss = stats[col]
        processor.mean_ = np.asarray(ss['mean'])
        processor.scale_ = np.asarray(ss['std'])
        processor.var_ = processor.scale_ ** 2
        processor.n_features_in_ = len(processor.mean_)
        processor.n_samples_seen_ = ss['n']"""
assert old in source;source=source.replace(old,new)

if 'init_value' in case_data:
 import gymnasium as gym
 class SyntheticGoalPairs:
  """Adapter supplies start/goal renders only; never claimed to contain trajectories."""
  column_names=['pixels','proprio','action']
  def __init__(self):
   self.rows={c['episode']:c for c in cases}
   self.env=gym.make('swm/TwoRoom-v1',init_value=case_data['init_value']).unwrapped
   self.env.reset(seed=9102030)
  def get_col_data(self,col):
   if col in ['ep_idx','episode_idx']:return np.array(list(self.rows))
   if col=='proprio':return np.array([c['start_xy'] for c in cases])
   if col=='action':return np.zeros((len(cases),2))
   raise KeyError(col)
  def load_chunk(self,episodes,starts,ends):
   result=[]
   for ep in episodes:
    c=self.rows[int(ep)];start=np.asarray(c['start_xy'],dtype=np.float32);goal=np.asarray(c['goals']['75']['xy'],dtype=np.float32)
    self.env._set_goal_state(goal.copy())
    self.env._set_state(start.copy());first=self.env.render().copy()
    self.env._set_state(goal.copy());last=self.env.render().copy()
    result.append({'pixels':torch.from_numpy(np.stack([first,last])).permute(0,3,1,2),'proprio':torch.from_numpy(np.stack([start,goal])),'action':torch.zeros(2,2)})
   return result
 original_world_init=swm.World.__init__
 def map_world_init(self,*a,**kw):
  kw['init_value']=case_data['init_value']
  original_world_init(self,*a,**kw)
 swm.World.__init__=map_world_init
 original_world_run=swm.World._run
 def checked_map_run(self,*a,**kw):
  audit=[]
  for i,c in enumerate(cases):
   env=self.envs.envs[i].unwrapped
   p=np.asarray(self.infos['pixels'][i]).reshape(-1,224,224,3)[-1]
   rendered=env.render()
   err=float(np.abs(p.astype(float)-rendered.astype(float)).max())
   state_error=float(np.linalg.norm(env.agent_position.numpy()-np.asarray(c['start_xy'])))
   assert err==0 and state_error<1e-5,(i,err,state_error)
   goal=np.asarray(c['goals']['75']['xy'],dtype=np.float32)
   saved=env.agent_position.numpy().copy()
   env._set_state(goal);goal_render=env.render()
   provided=np.asarray(self.infos['goal'][i]).reshape(-1,224,224,3)[-1]
   goal_error=float(np.abs(provided.astype(float)-goal_render.astype(float)).max())
   env._set_state(saved)
   assert goal_error==0,(i,goal_error)
   audit.append({'episode':c['episode'],'start_render_max_error':err,'goal_render_max_error':goal_error,'start_state_error':state_error})
  (out/'map_adapter_audit.json').write_text(json.dumps(audit,indent=2))
  return original_world_run(self,*a,**kw)
 swm.World._run=checked_map_run
 source=source.replace('dataset = get_dataset(cfg, cfg.eval.dataset_name)','dataset = SyntheticGoalPairs()')
 begin=source.index('    # sample the episodes and the starting indices')
 end=source.index('    world.set_policy(policy)',begin)
 source=source[:begin]+"""    eval_episodes = np.array([c['episode'] for c in cases])
    eval_start_idx = np.zeros(len(cases), dtype=np.int64)
"""+source[end:]

exec(compile(source,str(repo/'eval.py'),'exec'),globals())
