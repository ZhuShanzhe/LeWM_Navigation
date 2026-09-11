"""Freeze trajectory-group split, train-only statistics and cases before outcomes."""
import json,hashlib,time
from pathlib import Path
import numpy as np,h5py,hdf5plugin
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'strict_nav_20260910';D=R/'splits'
source=ROOT/'data/tworoom.h5'
if (D/'manifest.json').exists():raise RuntimeError('Frozen split already exists; do not overwrite')
with h5py.File(source,'r') as f:
 lengths=f['ep_len'][:];offsets=f['ep_offset'][:];pos=f['proprio'][:];act=f['action'][:]
 groups={}
 for ep,(o,l) in enumerate(zip(offsets,lengths)):
  h=hashlib.sha256(np.ascontiguousarray(pos[o:o+l]).tobytes()+np.ascontiguousarray(act[o:o+l]).tobytes()).hexdigest()
  groups.setdefault(h,[]).append(ep)
 keys=sorted(groups);rng=np.random.default_rng(9102026);rng.shuffle(keys)
 cuts=[0,int(.8*len(keys)),int(.9*len(keys)),len(keys)]
 splits={name:sorted(e for h in keys[cuts[j]:cuts[j+1]] for e in groups[h]) for j,name in enumerate(['train','validation','test'])}
 assert not set(splits['train']) & set(splits['test'])
 assert not set(splits['train']) & set(splits['validation'])
 assert not set(splits['validation']) & set(splits['test'])
 trainrows=np.concatenate([np.arange(offsets[e],offsets[e]+lengths[e]) for e in splits['train']])
 stats={}
 for key,arr in [('action',act),('proprio',pos)]:
  x=arr[trainrows];x=x[np.isfinite(x).all(axis=1)].astype(np.float64)
  stats[key]={'mean':x.mean(0).tolist(),'std':np.maximum(x.std(0,ddof=1),1e-8).tolist(),'n':len(x),'ddof':1}
 (D/'normalization.json').write_text(json.dumps(stats,indent=2))
 for name,n in [('validation',100),('test',300)]:
  r=np.random.default_rng(9102027 if name=='validation' else 9102028)
  pool=[e for e in splits[name] if lengths[e]>=81];chosen=r.choice(pool,min(n,len(pool)),replace=False)
  cases=[]
  for e in chosen:
   t=int(r.integers(0,int(lengths[e])-75));i=int(offsets[e])+t;x=pos[i]
   cases.append({'episode':int(e),'start_step':t,'start_xy':x.tolist(),'goals':{str(h):{'xy':pos[i+h].tolist(),'euclidean':float(np.linalg.norm(pos[i+h]-x)),'across_wall':bool((x[0]-112)*(pos[i+h,0]-112)<0),'initially_within_success':bool(np.linalg.norm(pos[i+h]-x)<16)} for h in [25,75]}})
  body={'split':name,'selection_seed':9102027 if name=='validation' else 9102028,'sampling':'one uniformly chosen eligible start per unique episode, length>=81; fixed before outcomes','cases':cases}
  (D/f'cases_{name}.json').write_text(json.dumps(body,indent=2))
  for j in range(0,len(cases),50):
   (D/f'cases_{name}_{j//50}.json').write_text(json.dumps({**body,'cases':cases[j:j+50]},indent=2))
 manifest={'created_unix':time.time(),'source':str(source),'source_bytes':source.stat().st_size,'split_seed':9102026,'grouping':'SHA256 of full proprio+action trajectories; exact duplicate trajectories stay together','n_episodes':len(lengths),'unique_trajectory_groups':len(groups),'episode_counts':{k:len(v) for k,v in splits.items()},'frame_counts':{k:int(lengths[v].sum()) for k,v in splits.items()},'episodes':splits,'all_episode_intersections_empty':True,'normalization_source':'train only; sample std ddof=1 shared with evaluation','caveat':'shared map and state support remain; strict episode exclusion, not unseen-map generalization','frozen_files_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('*.json')}}
 (D/'manifest.json').write_text(json.dumps(manifest,indent=2))
 print(json.dumps({k:v for k,v in manifest.items() if k not in ['episodes','frozen_files_sha256']},indent=2))
