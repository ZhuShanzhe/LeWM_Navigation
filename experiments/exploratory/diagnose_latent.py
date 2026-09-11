"""Episode-held-out latent diagnostics and fixed-candidate model/metric audit for TwoRooms."""
import os,sys,json,time,functools,subprocess
from pathlib import Path
import numpy as np,torch,h5py,hdf5plugin,gymnasium as gym
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error,r2_score,roc_auc_score
from scipy.stats import spearmanr
import stable_worldmodel as swm
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'round12h_20260909'
sys.path.insert(0,str(ROOT/'le-wm'))
tag=sys.argv[1];modelpath=sys.argv[2];out=R/'diagnostics'/tag;out.mkdir(parents=True,exist_ok=True)
# One serialized recovery within the next scheduled diagnostic, never alongside control timing.
if tag=='r12_noreg' and not (R/'diagnostics/trained3072/diagnostics.json').exists():
 print('RETRY_TRAINED3072_DIAGNOSTIC',flush=True)
 subprocess.run([sys.executable,__file__,'trained3072',str(ROOT/'data/checkpoints/r12_tw_s3072/weights_final.pt')],timeout=350,check=False)
if tag=='r12_dim64' and not (R/'diagnostics/base3000/diagnostics.json').exists():
 print('MATCHED_BASE3000_DIAGNOSTIC',flush=True)
 subprocess.run([sys.executable,__file__,'base3000',str(ROOT/'data/checkpoints/r12_tw_s3072/weights_step_3000.pt')],timeout=350,check=False)
rng=np.random.default_rng(2026);torch.manual_seed(2026);torch.set_num_threads(8)
model=swm.wm.utils.load_pretrained(modelpath).cuda().eval().requires_grad_(False)
hs=int(model.predictor.pos_embedding.shape[1])
if model.__class__.__module__=='jepa':model.rollout=functools.partial(model.rollout,history_size=hs)
mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None]
std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
def image(x):return (torch.as_tensor(np.asarray(x).copy(),device='cuda').permute(0,3,1,2).float()/255-mean)/std
@torch.no_grad()
def encode(x):
 return model.projector(model.encoder(image(x),interpolate_pos_encoding=True).last_hidden_state[:,0])
def corr(a,b):return [float(np.corrcoef(a[:,i],b[:,i])[0,1]) for i in range(a.shape[1])]
start=time.time()
with h5py.File(ROOT/'data/tworoom.h5','r') as f:
 offsets=f['ep_offset'][:];lengths=f['ep_len'][:];pos=f['proprio'][:];act=f['action'][:]
 amean=np.nanmean(act,axis=0);astd=np.nanstd(act,axis=0);astd=np.maximum(astd,1e-6)
 episodes=rng.permutation(len(lengths))[:700];groups={int(ep):0 if i<500 else 1 if i<600 else 2 for i,ep in enumerate(episodes)}
 indices=sorted({int(offsets[ep]+s) for ep in episodes for s in np.linspace(0,int(lengths[ep])-1,8,dtype=int)})
 epall=f['ep_idx'][:];split=np.array([groups[int(epall[i])] for i in indices]);z=[]
 for j in range(0,len(indices),128):
  z.append(encode(f['pixels'][indices[j:j+128]]).cpu().numpy())
 z=np.concatenate(z);y=pos[indices];np.savez_compressed(out/'embeddings.npz',z=z,position=y,indices=indices,split=split)
 print('ENCODED',len(z),flush=True)
 ym=y[split==0].mean(0);ys=y[split==0].std(0);ys=np.maximum(ys,1e-6);yn=(y-ym)/ys
 zm=z[split==0].mean(0);zs=np.maximum(z[split==0].std(0),1e-6);zn=(z-zm)/zs
 best=None
 for alpha in [.001,.01,.1,1,10,100]:
  reg=Ridge(alpha=alpha).fit(zn[split==0],yn[split==0]);err=mean_squared_error(yn[split==1],reg.predict(zn[split==1]))
  if best is None or err<best[0]:best=(err,alpha,reg)
 ridge=best[2];test=split==2;pred=ridge.predict(zn[test]);truth=yn[test]
 probes={'linear':{'alpha':best[1],'normalized_mse':float(mean_squared_error(truth,pred)),'pixel_mse':float(mean_squared_error(y[test],pred*ys+ym)),'r2':float(r2_score(truth,pred)),'pearson_per_axis':corr(truth,pred)}}
 mlp=MLPRegressor(hidden_layer_sizes=(128,128),max_iter=150,batch_size=256,early_stopping=True,random_state=2026).fit(zn[split==0],yn[split==0])
 pred=mlp.predict(zn[test]);probes['mlp']={'normalized_mse':float(mean_squared_error(truth,pred)),'r2':float(r2_score(truth,pred)),'pearson_per_axis':corr(truth,pred)}
 eig=np.maximum(np.linalg.eigvalsh(np.cov(z[split==0].T)),0);p=eig/max(eig.sum(),1e-12)
 representation={'mean_dimension_std':float(z.std(0).mean()),'effective_rank':float(np.exp(-(p*np.log(p+1e-12)).sum())),'variance_top2_fraction':float(eig[-2:].sum()/max(eig.sum(),1e-12))}
 # Compare teacher-forced and open-loop latent prediction on unseen probe episodes.
 selected=[int(e) for e in episodes[600:] if lengths[e]>=81][:64];errs={str(h):{'open_loop':[],'teacher_forced':[],'persistence':[],'action_embedding_corrupted':[]} for h in [1,3,5,10,15]}
 for ep in selected:
  i=int(offsets[ep]);frames=encode(f['pixels'][i:i+80:5]);a=np.nan_to_num(act[i:i+75]);aa=torch.as_tensor(((a-amean)/astd).reshape(1,15,10),device='cuda').float()
  ae=model.action_encoder(aa);hist=frames[:1].unsqueeze(0);shist=hist.clone()
  for t in range(15):
   nextz=model.predict(hist[:,-hs:],ae[:,max(0,t+1-hs):t+1])[:,-1:]
   shuffled=model.predict(shist[:,-hs:],ae[:,max(0,t+1-hs):t+1].flip(1).roll(5,dims=-1))[:,-1:]
   history_start=max(0,t+1-hs)
   teacher=model.predict(frames[history_start:t+1].unsqueeze(0),ae[:,history_start:t+1])[:,-1:]
   hist=torch.cat([hist,nextz],1);shist=torch.cat([shist,shuffled],1)
   if str(t+1) in errs:
    e=errs[str(t+1)];target=frames[t+1]
    for key,value in [('open_loop',nextz[0,0]),('teacher_forced',teacher[0,0]),('persistence',frames[0]),('action_embedding_corrupted',shuffled[0,0])]:e[key].append(float((value-target).square().mean()))
 rollout={h:{k:float(np.mean(v)) for k,v in e.items()} for h,e in errs.items()}
 # Synthetic jump detection: NOT the paper's complete violation-of-expectation protocol.
 valid=[];jump=[]
 for ep in selected:
  i=int(offsets[ep]);zz=encode(f['pixels'][i:i+6:5]);aa=torch.as_tensor(((np.nan_to_num(act[i:i+5])-amean)/astd).reshape(1,1,10),device='cuda').float()
  pred=model.predict(zz[:1].unsqueeze(0),model.action_encoder(aa))[0,0]
  valid.append(float((pred-zz[1]).square().mean()))
  far=np.where(np.linalg.norm(y-pos[i],axis=1)>100)[0]
  if len(far)==0:far=np.arange(len(z))
  target=torch.as_tensor(z[rng.choice(far)],device='cuda')
  jump.append(float((pred-target).square().mean()))
 surprise={'synthetic_jump_auroc':float(roc_auc_score([0]*len(valid)+[1]*len(jump),valid+jump)),'valid_mean':float(np.mean(valid)),'jump_mean':float(np.mean(jump)),'n_pairs':len(valid),'caveat':'Temporal jump surrogate; not full original VOE reproduction.'}
 print('PROBES_AND_ROLLOUT_DONE',flush=True)
 # Fixed candidates separate imperfect predicted endpoints from poor true-embedding scoring.
 env=gym.make('swm/TwoRoom-v1').unwrapped;env.reset(seed=2026);candidate_rows=[]
 for ep in selected[:24]:
  i=int(offsets[ep]);goal_idx=i+75;goal=pos[goal_idx]
  expert=np.nan_to_num(act[i:i+25]);cands=rng.normal(size=(24,25,2)).astype('float32')*.6
  cands[0]=expert;cands[1]=0;cands[2:12]=expert[None]+rng.normal(0,.35,size=(10,25,2))
  endpoints=[];endpoint_xy=[]
  for sequence in cands:
   env._set_state(pos[i]);env._set_goal_state(goal)
   for a in sequence:env.step(a)
   endpoints.append(env.render().copy());endpoint_xy.append(env.agent_position.detach().cpu().numpy().copy())
  truez=encode(np.stack(endpoints));goalz=encode(f['pixels'][goal_idx:goal_idx+1])[0]
  truecost=(truez-goalz).square().sum(-1).cpu().numpy()
  norm=((cands-amean)/astd).reshape(1,24,5,10)
  pix=image(f['pixels'][i:i+1])[:,None,None].expand(1,24,1,3,224,224)
  goalpix=image(f['pixels'][goal_idx:goal_idx+1])[:,None,None].expand_as(pix)
  info={'pixels':pix,'goal':goalpix,'action':torch.zeros(1,24,1,10,device='cuda')}
  predicted=model.get_cost(info,torch.as_tensor(norm,device='cuda')).detach().cpu().numpy().reshape(-1)
  coords=ridge.predict((truez.cpu().numpy()-zm)/zs)*ys+ym
  endpoint_xy=np.stack(endpoint_xy);distance=np.linalg.norm(endpoint_xy-goal,axis=1)
  across=(endpoint_xy[:,0]-112)*(goal[0]-112)<0
  waypoint=np.where(across,np.linalg.norm(endpoint_xy-np.array([112,49]),axis=1)+np.linalg.norm(goal-np.array([112,49])),distance)
  candidate_rows.append({'episode':ep,'predicted_vs_actual_latent_rank':float(spearmanr(predicted,truecost).statistic),'actual_latent_vs_actual_distance_rank':float(spearmanr(truecost,distance).statistic),'selected_actual_latent_cost':float(truecost[np.argmin(predicted)]),'best_actual_latent_cost':float(truecost.min()),'predicted_costs':predicted.tolist(),'actual_latent_costs':truecost.tolist(),'actual_endpoint_distances':distance.tolist(),'door_waypoint_proxy':waypoint.tolist(),'actual_latent_vs_waypoint_rank':float(spearmanr(truecost,waypoint).statistic),'endpoint_xy':endpoint_xy.tolist(),'goal_xy':goal.tolist()})
 env.close()
result={'model':modelpath,'seed':2026,'episode_split':{'train':500,'validation':100,'test':100},'probes':probes,'representation':representation,'rollout_mse':rollout,'surprise':surprise,'fixed_candidates':candidate_rows,'elapsed_seconds':time.time()-start,'limitations':['Probe episode split does not mean the world model never trained on those episodes.','Door-waypoint length is a geometry proxy, not an exact geodesic or a deployed policy.','Synthetic jump AUROC is not the original full physical violation protocol.']}
(out/'diagnostics.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='fixed_candidates'}),flush=True)
