"""Shared-candidate intervention audit. Simulator state only used in oracle diagnostics."""
import os,sys,time,json,functools
from pathlib import Path
import numpy as np,torch,h5py,hdf5plugin,gymnasium as gym
from scipy.stats import spearmanr
import stable_worldmodel as swm
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'strict_nav_20260910'
sys.path.insert(0,str(ROOT/'le-wm'))
tag,modelpath=sys.argv[1:3];out=R/'diagnostics'/tag;out.mkdir(parents=True,exist_ok=True)
manifest=json.loads((R/'splits/manifest.json').read_text())
case_data=json.loads((R/'splits/cases_test.json').read_text())
cases=case_data['cases'][:int(os.environ.get('LEWM_DIAG_CASES','100'))]
assert all(c['episode'] in manifest['episodes']['test'] for c in cases)
stats=json.loads((R/'splits/normalization.json').read_text());am=np.array(stats['action']['mean'],dtype=np.float32);ast=np.array(stats['action']['std'],dtype=np.float32)
torch.set_num_threads(4);torch.manual_seed(9102029);rng=np.random.default_rng(9102029)
model=swm.wm.utils.load_pretrained(modelpath).cuda().eval().requires_grad_(False);hs=int(model.predictor.pos_embedding.shape[1])
if model.__class__.__module__=='jepa':model.rollout=functools.partial(model.rollout,history_size=hs)
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
def image(x):return (torch.as_tensor(np.array(x),device='cuda').permute(0,3,1,2).float()/255-mean)/std
@torch.inference_mode()
def enc(x):return model.projector(model.encoder(image(x),interpolate_pos_encoding=True).last_hidden_state[:,0])
def rank(a,b):
 x=float(spearmanr(a,b).statistic);return x if np.isfinite(x) else None
env=gym.make('swm/TwoRoom-v1').unwrapped;env.reset(seed=9102029)
rows=[];start=time.time()
with h5py.File(ROOT/'data/tworoom.h5','r') as f:
 offsets=f['ep_offset'][:]
 for index,c in enumerate(cases):
  ep=c['episode'];i=int(offsets[ep])+c['start_step'];x=np.array(c['start_xy'],dtype=np.float32);g=np.array(c['goals']['75']['xy'],dtype=np.float32)
  px=f['pixels'][i:i+1];goalpx=f['pixels'][i+75:i+76]
  z=enc(px);zg=enc(goalpx)
  # CEM proposes a deployable pool without expert actions or geometry information.
  N=64;H=5
  pix=image(px)[:,None,None].expand(1,N,1,3,224,224);gp=image(goalpx)[:,None,None].expand_as(pix)
  def cost(actions):
   return model.get_cost({'pixels':pix,'goal':gp,'action':torch.zeros(1,N,1,10,device='cuda')},actions).reshape(-1)
  mu=torch.zeros(1,H,10,device='cuda');sigma=torch.ones_like(mu)
  with torch.inference_mode():
   for it in range(10):
    acts=mu[:,None]+sigma[:,None]*torch.randn(1,N,H,10,device='cuda');acts[:,0]=mu
    predicted=cost(acts);top=acts[:,predicted.topk(8,largest=False).indices];mu=top.mean(1);sigma=top.std(1)
   # Final mean is a returned control proposal; replace index0 and rescore fixed pool.
   acts[:,0]=mu;predicted=cost(acts).cpu().numpy()
  raw=acts.detach().cpu().numpy().reshape(N,25,2)*ast+am
  endpx=[];endxy=[];clipped_fraction=float((np.abs(raw)>1).mean())
  for sequence in raw:
   env._set_state(x.copy());env._set_goal_state(g.copy())
   for a in sequence:env.step(a)
   endxy.append(env.agent_position.numpy().copy());endpx.append(env.render().copy())
  truez=enc(np.stack(endpx));actual_latent=(truez-zg).square().sum(-1).cpu().numpy()
  xy=np.stack(endxy);dist=np.linalg.norm(xy-g,axis=1)
  # Conservative center-door route proxy, not exact shortest path.
  across=(xy[:,0]-112)*(g[0]-112)<0;door=np.array([112,49])
  route=np.where(across,np.linalg.norm(xy-door,axis=1)+np.linalg.norm(g-door),dist)
  choices={'predicted_latent':int(np.argmin(predicted)),'oracle_actual_latent':int(np.argmin(actual_latent)),'oracle_euclidean':int(np.argmin(dist)),'oracle_door_proxy':int(np.argmin(route))}
  row={'episode':ep,'start_step':c['start_step'],'across_wall':c['goals']['75']['across_wall'],'initial_euclidean':c['goals']['75']['euclidean'],'candidate_count':N,'action_clipping_fraction':clipped_fraction,'choices':choices,'predicted_costs':predicted.tolist(),'actual_latent_costs':actual_latent.tolist(),'endpoint_distance':dist.tolist(),'door_route_proxy':route.tolist(),'endpoint_xy':xy.tolist(),'rank_pred_vs_actual_latent':rank(predicted,actual_latent),'rank_actual_latent_vs_distance':rank(actual_latent,dist),'rank_actual_latent_vs_door_proxy':rank(actual_latent,route),'selection_regret_actual_latent':float(actual_latent[choices['predicted_latent']]-actual_latent.min())}
  # Closed-loop one-block intervention from identical state: selected sequence prefix only.
  outcome={}
  for label,k in choices.items():
   env._set_state(x.copy());env._set_goal_state(g.copy())
   success=False
   for a in raw[k,:5]:
    _,_,done,_,_=env.step(a);success=success or done
   y=env.agent_position.numpy().copy()
   outcome[label]={'xy_after_5':y.tolist(),'distance_after_5':float(np.linalg.norm(y-g)),'success_within_5':bool(success)}
  row['prefix_interventions']=outcome
  # True-action multi-step prediction versus teacher-forced correction on the held-out episode.
  frames=enc(f['pixels'][i:i+80:5]);a=np.nan_to_num(f['action'][i:i+75]);aa=torch.as_tensor(((a-am)/ast).reshape(1,15,10),device='cuda')
  with torch.inference_mode():
   ae=model.action_encoder(aa);hist=frames[:1][None];errors={}
   for t in range(15):
    nz=model.predict(hist[:,-hs:],ae[:,max(0,t+1-hs):t+1])[:,-1:]
    tf=model.predict(frames[max(0,t+1-hs):t+1][None],ae[:,max(0,t+1-hs):t+1])[:,-1:]
    hist=torch.cat([hist,nz],1)
    if t+1 in [1,3,5,10,15]:
     errors[str(t+1)]={'open_loop':float((nz[0,0]-frames[t+1]).square().mean()),'teacher_forced':float((tf[0,0]-frames[t+1]).square().mean())}
  row['logged_action_errors']=errors;rows.append(row)
  if index%10==0:
   (out/'progress.json').write_text(json.dumps({'completed':len(rows),'total':len(cases),'elapsed_seconds':time.time()-start}))
   print('AUDIT',len(rows),flush=True)
env.close()
payload={'model':modelpath,'test_cases':len(rows),'candidate_seed':9102029,'normalization':'train only','rows':rows,'elapsed_seconds':time.time()-start,'limitations':['One-block choice interventions are not full closed-loop oracle policies.','Candidate set comes from finite CEM proposals; oracle choice does not prove globally optimal search.','Door center route is a geometry proxy, not exact geodesic.','Oracle scoring uses hidden simulator state and cannot be ranked as a visual baseline.','No outcome-dependent test-case filtering.']}
(out/'candidate_audit.json').write_text(json.dumps(payload,indent=2));print('AUDIT_COMPLETE',len(rows),flush=True)
