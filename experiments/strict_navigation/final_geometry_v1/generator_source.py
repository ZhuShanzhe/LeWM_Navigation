"""Freeze controlled TwoRoom geometry cases; verify dynamics and expert solvability."""
import json,time,hashlib
from pathlib import Path
import numpy as np,torch,gymnasium as gym
import stable_worldmodel
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');D=R/'final_geometry_v1'
torch.set_num_threads(1)
specs=[]
for split,centers in [('final_geometry',[73,169])]:
 for axis in [1,0]:
  for c in centers:specs.append({'id':f'{split}_axis{axis}_door{c}','split':split,'axis':axis,'door_center':c,'init_value':{'wall.axis':axis,'door.position':[c,c,c],'door.number':1,'door.size':[14,14,14]}})
if (D/'catalog.json').exists():raise RuntimeError('Frozen catalog exists; do not overwrite')
rng=np.random.default_rng(9106001)
# 50 matched Euclidean-distance cases per side relationship. These are generated goals,
# not future trajectory offsets. Same coordinate pairs are reused across door locations.
base=[]
for across in [False,True]:
 while sum(x['across'] is across for x in base)<50:
  x=np.array([rng.uniform(30,92),rng.uniform(30,194)])
  g=np.array([rng.uniform(132,194) if across else rng.uniform(30,92),rng.uniform(30,194)])
  dist=np.linalg.norm(x-g)
  if not 75<=dist<=125:continue
  if rng.random()<.5:x[0]=224-x[0];g[0]=224-g[0]
  base.append({'start':x.tolist(),'goal':g.tolist(),'across':across})
audit=[]
def expert_action(env,g):
 x=env.agent_position.numpy();axis=0 if env.wall_axis==1 else 1;other=1-axis;sgn=1 if g[axis]>112 else -1
 waypoint=g.copy()
 if (x[axis]-112)*(g[axis]-112)<0 or abs(x[axis]-112)<15:
  door=float(env.door_positions[0]);source_side=-sgn
  # Approach a point clear of the wall, align, cross to the far side, then target.
  if abs(x[other]-door)>3 and abs(x[axis]-112)>14:
   waypoint=x.copy();waypoint[axis]=112+source_side*18;waypoint[other]=door
  else:
   waypoint=x.copy();waypoint[axis]=112+sgn*18;waypoint[other]=door
 delta=waypoint-x
 return np.clip(delta/5,-1,1).astype(np.float32)
for spec in specs:
 env=gym.make('swm/TwoRoom-v1',init_value=spec['init_value']).unwrapped;env.reset(seed=9106001)
 rows=[];successes=0;maxsteps=0
 for i,b in enumerate(base):
  x=np.array(b['start'],dtype=np.float32);g=np.array(b['goal'],dtype=np.float32)
  if spec['axis']==0:x=x[::-1].copy();g=g[::-1].copy()
  env._set_state(x.copy());env._set_goal_state(g.copy())
  first=env.render();env.step(np.zeros(2,dtype=np.float32));noop_delta=float(np.linalg.norm(env.agent_position.numpy()-x))
  assert noop_delta<1e-6
  # Crossing at the door must work; crossing a distant wall segment must fail.
  env._set_state(x.copy());success=False
  for n in range(150):
   _,_,done,_,_=env.step(expert_action(env,g))
   if done:success=True;break
  successes+=int(success);maxsteps=max(maxsteps,n+1)
  dist=float(np.linalg.norm(x-g))
  row={'episode':i,'start_step':0,'start_xy':x.tolist(),'goals':{'75':{'xy':g.tolist(),'euclidean':dist,'across_wall':b['across'],'initially_within_success':False}},'oracle_reference_solved':success,'oracle_reference_steps':n+1}
  rows.append(row)
 axis=0 if spec['axis']==1 else 1;other=1-axis;door=spec['door_center']
 def cross(location):
  p=np.array([80.,80.],dtype=np.float32);p[axis]=90;p[other]=location;env._set_state(p)
  a=np.zeros(2,dtype=np.float32);a[axis]=1
  for _ in range(10):env.step(a)
  return float(env.agent_position[axis])
 assert cross(door)>124
 blocked=30 if abs(door-30)>30 else 194
 assert cross(blocked)<112
 assert successes==100,(spec['id'],successes)
 body={**spec,'selection_seed':9106001,'task':'synthetic direct-goal navigation; key 75 is adapter compatibility, not logged horizon','distance_range':[75,125],'cases':rows}
 (D/(spec['id']+'.json')).write_text(json.dumps(body,indent=2))
 for k in [0,1]:(D/(spec['id']+f'_b{k}.json')).write_text(json.dumps({**body,'cases':rows[k*50:(k+1)*50]},indent=2))
 audit.append({'map':spec['id'],'oracle_reference_successes':successes,'n':100,'max_reference_steps':maxsteps,'noop_state_consistency':True,'wall_blocks_and_door_passes':True})
 env.close()
(D/'catalog.json').write_text(json.dumps({'created_unix':time.time(),'maps':specs,'audit':audit,'geometry_note':'two rooms, one door; changed geometry and wall orientation, NOT new connectivity topology','train_test_geometry_disjoint':True,'cases_reused_across_maps':True,'generated_case_count':100*len(specs)},indent=2))
print('MAP_CATALOG_READY',len(specs),flush=True)
