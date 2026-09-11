import json,time
from pathlib import Path
import numpy as np,h5py,hdf5plugin,torch,gymnasium as gym
import stable_worldmodel as swm
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');torch.set_num_threads(1)
results={}
for variant in ['single','multi']:
 out=R/'collection_smoke_v2'/variant;summary=json.loads((out/'summary.json').read_text());audit=json.loads((out/'episode_audit.json').read_text())
 specs={m['id']:m for m in json.loads((R/'maps/catalog.json').read_text())['maps']}
 f=h5py.File(summary['dataset'],'r');renders=0;transitions=0
 for row in audit:
  ep=row['episode'];off=int(f['ep_offset'][ep]);length=int(f['ep_len'][ep])
  env=gym.make('swm/TwoRoom-v1',init_value=specs[row['map']]['init_value']).unwrapped;env.reset(seed=123)
  for t in [0,length//2,length-2]:
   env._set_state(f['proprio'][off+t]);render=env.render()
   assert np.array_equal(render,f['pixels'][off+t]);renders+=1
   env.step(f['action'][off+t])
   assert np.max(np.abs(env.agent_position.numpy()-f['proprio'][off+t+1]))<1e-5;transitions+=1
  env.close()
 stats=json.loads((out/'normalization.json').read_text())
 train=np.isin(f['ep_idx'][:],json.loads((out/'manifest.json').read_text())['episodes']['train'])
 for key in ['action','proprio']:
  values=f[key][:][train].astype(np.float64)
  assert np.allclose(values.mean(0),stats[key]['mean'],atol=1e-10)
  assert np.allclose(values.std(0,ddof=1),stats[key]['std'],atol=1e-10)
 f.close()
 ds=swm.data.HDF5Dataset(str(Path(summary['dataset']).with_suffix('')),num_steps=2,frameskip=5,keys_to_load=['pixels','action','proprio'],keys_to_cache=['action','proprio'])
 sample=ds[0]
 results[variant]={'render_matches':renders,'transition_matches':transitions,'normalization_verified':True,'loader_clips':len(ds),'sample_shapes':{k:list(v.shape) for k,v in sample.items() if hasattr(v,'shape')},'passed':True}
(R/'collection_smoke_v2/audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
