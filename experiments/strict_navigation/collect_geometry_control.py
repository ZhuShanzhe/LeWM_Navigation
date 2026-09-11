"""Matched collector controls: same frames, policy mixture and seeds; change training geometry only."""
import os,json,time,hashlib,sys
from pathlib import Path
import numpy as np,torch,h5py,hdf5plugin,gymnasium as gym
import stable_worldmodel
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
variant=sys.argv[1];assert variant in ['single','multi']
smoke=os.environ.get('LEWM_COLLECTION_SMOKE')=='1'
out=R/('collection_smoke_v2' if smoke else 'collections')/variant
out.mkdir(parents=True,exist_ok=True)
target=(out/'dataset.h5') if smoke else ROOT/f'data/nav_{variant}_control.h5'
assert not target.exists(),f'Refuse to overwrite {target}'
partial=target.with_suffix('.partial.h5');assert not partial.exists(),f'Review incomplete collection {partial}'
torch.set_num_threads(1);starttime=time.time()
source=json.loads((R/'splits/manifest.json').read_text());catalog=json.loads((R/'maps/catalog.json').read_text())
with h5py.File(ROOT/'data/tworoom.h5','r') as src:original_lens=src['ep_len'][:]
ids=source['episodes']['train']+source['episodes']['validation']
splits=['train']*8000+['validation']*1000
if smoke:ids=source['episodes']['train'][:16]+source['episodes']['validation'][:8];splits=['train']*16+['validation']*8
lengths=original_lens[ids].astype(np.int32)
offset=np.r_[0,np.cumsum(lengths[:-1])].astype(np.int64);total=int(lengths.sum())
single=next(m for m in catalog['maps'] if m['id']=='train_axis1_door49')
specs=[];behaviors=[]
counts={'train':0,'validation':0}
for split in splits:
 eligible=[m for m in catalog['maps'] if m['split']==split]
 spec=single if variant=='single' else eligible[counts[split]%len(eligible)]
 specs.append(spec);behaviors.append('random' if (counts[split]//len(eligible))%2==0 else 'waypoint');counts[split]+=1
def point(rng,axis,side=None):
 p=np.array([rng.uniform(30,92),rng.uniform(30,194)],dtype=np.float32)
 if side is None:side=-1 if rng.random()<.5 else 1
 if side>0:p[0]=224-p[0]
 return p if axis==1 else p[::-1].copy()
def waypoint_action(env,goal):
 x=env.agent_position.numpy();axis=0 if env.wall_axis==1 else 1;other=1-axis;sgn=1 if goal[axis]>112 else -1
 wp=goal.copy()
 if (x[axis]-112)*(goal[axis]-112)<0 or abs(x[axis]-112)<15:
  door=float(env.door_positions[0])
  if abs(x[other]-door)>3 and abs(x[axis]-112)>14:
   wp=x.copy();wp[axis]=112-sgn*18;wp[other]=door
  else:wp=x.copy();wp[axis]=112+sgn*18;wp[other]=door
 return np.clip((wp-x)/5,-1,1)
sums={k:np.zeros(2,np.float64) for k in ['action','proprio']}
squares={k:np.zeros(2,np.float64) for k in sums};ntrain=0;episode_audits=[];hashes=set()
envs={}
with h5py.File(partial,'w') as f:
 f.attrs['protocol']='matched collector v1; 50% persistent random and 50% privileged waypoint-guided episodes; no evaluation cases'
 f.attrs['variant']=variant;f.attrs['smoke']=smoke
 f.create_dataset('ep_len',data=lengths);f.create_dataset('ep_offset',data=offset)
 f.create_dataset('pixels',(total,224,224,3),dtype=np.uint8,chunks=(1,224,224,3),**hdf5plugin.Blosc(cname='lz4',clevel=5,shuffle=hdf5plugin.Blosc.SHUFFLE))
 for name in ['action','proprio']:f.create_dataset(name,(total,2),dtype=np.float32)
 for name in ['ep_idx','step_idx','id']:f.create_dataset(name,(total,),dtype=np.int64)
 for ep,(length,off,spec,split) in enumerate(zip(lengths,offset,specs,splits)):
  rng=np.random.default_rng(9102100+ep);key=spec['id']
  if key not in envs:envs[key]=gym.make('swm/TwoRoom-v1',init_value=spec['init_value']).unwrapped
  env=envs[key];env.reset(seed=9102100+ep)
  x=point(rng,spec['axis']);env._set_state(x)
  axis=0 if spec['axis']==1 else 1;side=1 if x[axis]>112 else -1
  goal=point(rng,spec['axis'],-side if rng.random()<.5 else side);env._set_goal_state(goal)
  actions=[];states=[];pixels=[];cross=0;stalls=0;mode=behaviors[ep]
  for t in range(int(length)):
   x=env.agent_position.numpy().copy();states.append(x);pixels.append(env.render().copy())
   if mode=='waypoint':
    if np.linalg.norm(x-goal)<16:
     side=1 if x[axis]>112 else -1
     goal=point(rng,spec['axis'],-side if rng.random()<.5 else side);env._set_goal_state(goal)
    action=np.clip(waypoint_action(env,goal)+rng.normal(0,.12,2),-1,1).astype(np.float32)
   else:
    if t%5==0:held=rng.uniform(-1,1,2).astype(np.float32)
    action=held.copy()
   actions.append(action);env.step(action)
   y=env.agent_position.numpy()
   cross+=int((x[axis]-112)*(y[axis]-112)<0);stalls+=int(np.linalg.norm(x-y)<.5)
  acts=np.stack(actions);pos=np.stack(states);px=np.stack(pixels);sl=slice(int(off),int(off+length))
  digest=hashlib.sha256(pos.tobytes()+acts.tobytes()+key.encode()).hexdigest()
  assert digest not in hashes,'Duplicate generated trajectory';hashes.add(digest)
  f['pixels'][sl]=px;f['action'][sl]=acts;f['proprio'][sl]=pos
  f['ep_idx'][sl]=ep;f['step_idx'][sl]=np.arange(length);f['id'][sl]=np.arange(off,off+length)
  if split=='train':
   ntrain+=int(length)
   for name,arr in [('action',acts),('proprio',pos)]:
    arr=arr.astype(np.float64);sums[name]+=arr.sum(0);squares[name]+=(arr*arr).sum(0)
  episode_audits.append({'episode':ep,'split':split,'map':key,'length':int(length),'behavior':mode,'wall_midline_crossings':cross,'stalled_steps':stalls,'sha256':digest})
  if ep%100==0:
   f.flush();(out/'progress.json').write_text(json.dumps({'episode':ep+1,'total_episodes':len(lengths),'frames_done':int(off+length),'total_frames':total,'elapsed_seconds':time.time()-starttime}))
   print('COLLECT',variant,ep+1,len(lengths),flush=True)
 f.attrs['complete']=True
partial.replace(target)
stats={name:{'n':ntrain,'mean':(sums[name]/ntrain).tolist(),'std':np.sqrt(np.maximum((squares[name]-sums[name]**2/ntrain)/(ntrain-1),1e-16)).tolist()} for name in sums}
manifest={'dataset':str(target),'episodes':{'train':[i for i,s in enumerate(splits) if s=='train'],'validation':[i for i,s in enumerate(splits) if s=='validation'],'test':[]},
 'episode_counts':{'train':splits.count('train'),'validation':splits.count('validation'),'test':0},
 'protocol':'controlled collector, not original dataset reproduction; test layouts excluded from collection',
 'variant':variant,'train_frames':ntrain,'total_frames':total,'geometry_train':sorted({m['id'] for m,s in zip(specs,splits) if s=='train'}),'geometry_validation':sorted({m['id'] for m,s in zip(specs,splits) if s=='validation'})}
(out/'normalization.json').write_text(json.dumps(stats,indent=2))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
(out/'episode_audit.json').write_text(json.dumps(episode_audits,indent=2))
summary={'complete':True,'smoke':smoke,'variant':variant,'dataset':str(target),'episodes':len(lengths),'frames':total,'train_frames':ntrain,'elapsed_seconds':time.time()-starttime,'dataset_bytes':target.stat().st_size,
 'extra_supervision':'50% collection episodes use map geometry waypoint controller; model inputs still only images/actions',
 'geometry_counts':{k:sum(m['id']==k for m in specs) for k in sorted(set(m['id'] for m in specs))},
 'wall_midline_crossings':sum(x['wall_midline_crossings'] for x in episode_audits),'behavior_by_geometry':{k:{b:sum(x['map']==k and x['behavior']==b for x in episode_audits) for b in ['random','waypoint']} for k in sorted(set(x['map'] for x in episode_audits))}}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('COLLECTION_COMPLETE',json.dumps(summary),flush=True)
for env in envs.values():env.close()
