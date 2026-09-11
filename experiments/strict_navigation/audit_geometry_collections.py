"""Audit both completed controlled collections before authorizing their training."""
import json,hashlib
from pathlib import Path
import numpy as np,h5py,hdf5plugin,torch,gymnasium as gym
import stable_worldmodel as swm
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');torch.set_num_threads(1)
specs={m['id']:m for m in json.loads((R/'maps/catalog.json').read_text())['maps']}
reports={};lengths_by_variant={};audit_rows={}
for variant in ['single','multi']:
 out=R/'collections'/variant
 summary=json.loads((out/'summary.json').read_text());manifest=json.loads((out/'manifest.json').read_text())
 stats=json.loads((out/'normalization.json').read_text());episodes=json.loads((out/'episode_audit.json').read_text())
 assert summary['complete'] and not summary['smoke'] and summary['episodes']==9000
 assert len(episodes)==9000 and len({e['sha256'] for e in episodes})==9000
 assert manifest['episode_counts']=={'train':8000,'validation':1000,'test':0}
 assert not set(manifest['episodes']['train'])&set(manifest['episodes']['validation'])
 assert not manifest['episodes']['test']
 assert not any(specs[e['map']]['split']=='test' for e in episodes)
 filename=Path(summary['dataset'])
 with h5py.File(filename,'r') as f:
  assert bool(f.attrs['complete'])
  lengths=f['ep_len'][:];offset=f['ep_offset'][:];epidx=f['ep_idx'][:]
  lengths_by_variant[variant]=lengths
  assert len(lengths)==9000 and int(lengths.sum())==829038
  assert np.array_equal(epidx,np.repeat(np.arange(9000),lengths))
  assert np.array_equal(offset,np.r_[0,np.cumsum(lengths[:-1])])
  assert f['pixels'].shape==(829038,224,224,3)
  data={k:f[k][:] for k in ['action','proprio']}
  assert all(np.isfinite(x).all() for x in data.values());assert np.abs(data['action']).max()<=1
  train=np.isin(epidx,manifest['episodes']['train']);assert train.sum()==736684
  for k,v in data.items():
   values=v[train].astype(np.float64)
   assert np.allclose(values.mean(0),stats[k]['mean'],atol=1e-9)
   assert np.allclose(values.std(0,ddof=1),stats[k]['std'],atol=1e-9)
  counts={};coverage={}
  for ep,e in enumerate(episodes):
   assert e['episode']==ep and e['length']==lengths[ep]
   assert e['split']==('train' if ep<8000 else 'validation')
   sl=slice(int(offset[ep]),int(offset[ep]+lengths[ep]))
   digest=hashlib.sha256(data['proprio'][sl].tobytes()+data['action'][sl].tobytes()+e['map'].encode()).hexdigest()
   assert digest==e['sha256']
   key=e['split']+'/'+e['map'];counts.setdefault(key,{'random':0,'waypoint':0});counts[key][e['behavior']]+=1
   agg=coverage.setdefault(key,{'episodes':0,'episodes_crossed_midline':0,'midline_crossings':0,'stalled_steps':0,'frames':0,'cells':set()})
   agg['episodes']+=1;agg['episodes_crossed_midline']+=int(e['wall_midline_crossings']>0);agg['midline_crossings']+=e['wall_midline_crossings'];agg['stalled_steps']+=e['stalled_steps'];agg['frames']+=e['length']
   agg['cells'].update(map(tuple,np.floor(data['proprio'][sl]/14).astype(int)))
  assert all(v['random']==v['waypoint'] for v in counts.values())
  selected=np.random.default_rng(9102120).choice(9000,128,replace=False);envs={};transitions=0;renders=0
  for ep in selected:
   row=episodes[ep];key=row['map']
   if key not in envs:
    envs[key]=gym.make('swm/TwoRoom-v1',init_value=specs[key]['init_value']).unwrapped;envs[key].reset(seed=9102120)
   env=envs[key];off=int(offset[ep]);ln=int(lengths[ep])
   for t in [0,ln//2,ln-2]:
    env._set_state(data['proprio'][off+t])
    assert np.array_equal(env.render(),f['pixels'][off+t]);renders+=1
    env.step(data['action'][off+t]);assert np.max(np.abs(env.agent_position.numpy()-data['proprio'][off+t+1]))<1e-5;transitions+=1
  for env in envs.values():env.close()
 for v in coverage.values():v['grid_cells_14px']=len(v.pop('cells'))
 ds=swm.data.HDF5Dataset(str(filename.with_suffix('')),num_steps=2,frameskip=5,keys_to_load=['pixels','action','proprio'],keys_to_cache=['action','proprio'])
 assert len(ds)==748038
 reports[variant]={'passed':True,'dataset':str(filename),'bytes':filename.stat().st_size,'episodes':9000,'train_frames':736684,'total_frames':829038,'loader_clips':len(ds),
  'render_matches':renders,'transition_matches':transitions,'behavior_counts':counts,'coverage':coverage,
  'manifest_sha256':hashlib.sha256((out/'manifest.json').read_bytes()).hexdigest(),'normalization_sha256':hashlib.sha256((out/'normalization.json').read_bytes()).hexdigest()}
 audit_rows[variant]=episodes
assert np.array_equal(lengths_by_variant['single'],lengths_by_variant['multi'])
assert [e['behavior'] for e in audit_rows['single']]==[e['behavior'] for e in audit_rows['multi']]
assert [e['split'] for e in audit_rows['single']]==[e['split'] for e in audit_rows['multi']]
result={'passed':True,'paired_length_and_behavior_protocol':True,'variants':reports}
(R/'collections/audit.json').write_text(json.dumps(result,indent=2))
print('CONTROLLED_COLLECTION_AUDIT_PASSED',json.dumps({k:{x:v[x] for x in ['passed','episodes','render_matches','transition_matches']} for k,v in reports.items()}))
