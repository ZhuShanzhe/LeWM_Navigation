import json,hashlib,numpy as np,h5py,hdf5plugin
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909');D=R/'focused';D.mkdir(exist_ok=True)
with h5py.File(R.parent/'data/tworoom.h5','r') as f:
 lengths=f['ep_len'][:];offsets=f['ep_offset'][:];pos=f['proprio'][:]
 rng=np.random.default_rng(9172026);pool=np.where(lengths>=81)[0];chosen=rng.choice(pool,150,replace=False)
 rows=[]
 for ep in chosen:
  start=int(rng.integers(0,int(lengths[ep])-75));i=int(offsets[ep])+start;x=pos[i]
  row={'episode':int(ep),'start_step':start,'start_xy':x.tolist(),'goals':{}}
  for h in [25,75]:
   y=pos[i+h];distance=float(np.linalg.norm(y-x));row['goals'][str(h)]={'xy':y.tolist(),'euclidean':distance,'across_wall':bool((x[0]-112)*(y[0]-112)<0),'initially_within_success':distance<16}
  rows.append(row)
 for k,seed in enumerate([42,43,44]):
  body={'selection_seed':9172026,'batch':k,'eval_seed':seed,'pool_episodes':len(pool),'sampling':'150 unique episodes length>=81, one uniform eligible start per episode; split into 3 disjoint batches; no outcome-based selection','wall_axis':1,'wall_position':112,'success_distance_strict_less_than':16,'cases':rows[k*50:(k+1)*50]}
  (D/f'cases_s{seed}.json').write_text(json.dumps(body,indent=2))
 (D/'case_manifest.json').write_text(json.dumps({'unique_episodes':len(set(int(e) for e in chosen)),'n':len(rows),'source':'author released tworoom.h5','caveat':'same known layout and training-distribution dataset, not unseen-map generalization','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in D.glob('cases_*.json')}},indent=2))
print('CASES_READY',len(rows),sum(r['goals']['75']['across_wall'] for r in rows))
