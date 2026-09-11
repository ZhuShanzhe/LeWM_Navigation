"""CPU-only finite topology feasibility audit. Never train or select a model here."""
import json,time,hashlib,collections,os
from pathlib import Path
import numpy as np,torch
from PIL import Image,ImageDraw
from topology_env import TopologyEnv,topology_specs
from stable_worldmodel.envs.two_room.env import TwoRoomEnv
torch.set_num_threads(1)
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');D=R/os.environ.get('LEWM_TOPOLOGY_AUDIT_TAG','topology_dev_v2');D.mkdir(exist_ok=True)
assert not (D/'catalog.json').exists(),'Frozen catalog exists; do not overwrite'
rng=np.random.default_rng(9104101);catalog=[];summary=[];panels=[]
def path_between(spec,start,end):
 adj={i:[] for i in range(len(spec['room_centers']))}
 for a,b in spec['edges']:adj[a].append(b);adj[b].append(a)
 q=collections.deque([[start]]);seen={start}
 while q:
  p=q.popleft()
  if p[-1]==end:return p
  for z in adj[p[-1]]:
   if z not in seen:seen.add(z);q.append(p+[z])
 raise AssertionError('Disconnected room graph')
def point(spec,room):
 x0,x1,y0,y1=spec['room_bounds'][room]
 return np.array([rng.uniform(x0+22,x1-22),rng.uniform(y0+22,y1-22)],np.float32)
def reference(env,spec,start,goal,rs,rg):
 rooms=path_between(spec,rs,rg);waypoints=[]
 if len(rooms)>1:
  waypoints.append(np.array(spec['room_centers'][rs]))
  for a,b in zip(rooms,rooms[1:]):
   gate=next(g for g in spec['gates'] if set(g['edge'])=={a,b})
   waypoints.extend([np.array(gate['xy']),np.array(spec['room_centers'][b])])
 waypoints.append(goal);env._set_state(start.copy());env._set_goal_state(goal.copy())
 trace=[start.tolist()];steps=0;success=False
 for target in waypoints:
  while np.linalg.norm(env.agent_position.numpy()-target)>.05:
   cur=env.agent_position.numpy().copy()
   action=np.clip((target-cur)/5,-1,1).astype(np.float32)
   _,_,success,_,_=env.step(action);steps+=1;trace.append(env.agent_position.numpy().tolist())
   assert env.valid(env.agent_position.numpy())
   if success or steps>=150:return bool(success),steps,trace
 return success,steps,trace

for spec in topology_specs():
 env=TopologyEnv(spec);env.reset(seed=9104101);nrooms=len(spec['room_centers']);edge_set={tuple(e) for e in spec['edges']}
 degrees=[sum(i in e for e in spec['edges']) for i in range(nrooms)]
 diameter=max(len(path_between(spec,a,b))-1 for a in range(nrooms) for b in range(nrooms))
 # Spatial room adjacency must equal declared graph; prevents accidental shortcuts.
 observed=[]
 for a in range(nrooms):
  for b in range(a+1,nrooms):
   aa=np.array(spec['room_centers'][a]);bb=np.array(spec['room_centers'][b])
   ax=abs(aa[0]-bb[0])<1e-6;ay=abs(aa[1]-bb[1])<1e-6
   if not(ax or ay):continue
   middle=(aa+bb)/2
   bounds=spec['room_bounds']
   adjacent=(ax and abs(bounds[a][3]-bounds[b][2])<1e-6) or (ay and abs(bounds[a][1]-bounds[b][0])<1e-6)
   if adjacent:
    if tuple((a,b)) in edge_set:
     g=next(g for g in spec['gates'] if set(g['edge'])=={a,b})
     mid=np.array(g['xy']);normal=np.array(g['normal'])
     assert env.clear(mid-normal*20,mid+normal*20)
     observed.append((a,b))
    else:assert not env.clear(aa,bb)
 assert set(observed)==edge_set
 # Every declared doorway passes actual environment dynamics in both directions.
 gate_checks=0
 for gate in spec['gates']:
  mid=np.array(gate['xy']);normal=np.array(gate['normal'])
  for sign in [-1,1]:
   start=mid-sign*20*normal;goal=mid+sign*20*normal
   env._set_state(start);env._set_goal_state(goal)
   for _ in range(8):env.step((sign*normal).astype(np.float32))
   assert np.linalg.norm(env.agent_position.numpy()-goal)<1e-5;gate_checks+=1
 # Sample hard wall segments where both probe endpoints are valid.
 blocks=0
 for rect in spec['rectangles']:
  x0,x1,y0,y1=rect;mid=np.array([(x0+x1)/2,(y0+y1)/2]);normal=np.array([1,0]) if x1-x0<=y1-y0 else np.array([0,1])
  start=mid-20*normal;goal=mid+20*normal
  if env.valid(start) and env.valid(goal):
   env._set_state(start);env._set_goal_state(goal)
   for _ in range(10):env.step(normal.astype(np.float32))
   assert np.dot(env.agent_position.numpy()-mid,normal)<0;blocks+=1
 assert blocks>0
 # Raster mask comes from the same rectangles used for conservative inflated collision.
 y,x=np.mgrid[:224,:224];expected=np.zeros((224,224),bool)
 for x0,x1,y0,y1 in spec['rectangles']:expected|=(x>=x0)&(x<=x1)&(y>=y0)&(y<=y1)
 expected[:,10:14]=1;expected[:,210:214]=1;expected[10:14,:]=1;expected[210:214,:]=1
 assert np.array_equal(expected,env._wall_and_door_masks()[0].numpy())
 random_steps=0
 for room in range(nrooms):
  env._set_state(point(spec,room));env._set_goal_state(point(spec,room))
  for _ in range(200):
   env.step(rng.uniform(-1,1,2).astype(np.float32));assert env.valid(env.agent_position.numpy());random_steps+=1
 rows=[];traces=[]
 for i in range(100):
  same=i<20
  for attempt in range(10000):
   rs=int(rng.integers(nrooms));rg=rs if same else int(rng.choice([j for j in range(nrooms) if j!=rs]))
   start=point(spec,rs);goal=point(spec,rg);dist=float(np.linalg.norm(start-goal))
   if dist>=35:break
  else:raise AssertionError('case sampling exhausted')
  env._set_state(start);env._set_goal_state(goal);im0=env.render().copy();env.step(np.zeros(2,np.float32))
  assert np.array_equal(im0,env.render()) and np.array_equal(start,env.agent_position.numpy())
  env._set_state(goal);assert np.array_equal(env.render().transpose(2,0,1),env._target_img.numpy())
  success,steps,trace=reference(env,spec,start,goal,rs,rg);assert success,(spec['id'],i,steps)
  row={'episode':i,'start_step':0,'start_xy':start.tolist(),'goals':{'75':{'xy':goal.tolist(),'euclidean':dist,'different_room':not same,'initially_within_success':False}},'start_room':rs,'goal_room':rg,'room_graph_hops':len(path_between(spec,rs,rg))-1,'oracle_reference_solved':True,'oracle_reference_steps':steps,'reference_path_length_not_geodesic':float(np.linalg.norm(np.diff(np.array(trace),axis=0),axis=1).sum())}
  rows.append(row);traces.append(trace)
 body={'id':spec['id'],'split':'development','topology_spec':spec,'selection_seed':9104101,'task':'full-observation custom topology stress; 75 only adapter key, not trajectory offset','cases':rows,'not_final_untouched_test':True}
 (D/(spec['id']+'.json')).write_text(json.dumps(body,indent=2))
 (D/(spec['id']+'_reference_traces.json')).write_text(json.dumps(traces))
 for b in range(2):(D/(spec['id']+f'_b{b}.json')).write_text(json.dumps({**body,'cases':rows[b*50:(b+1)*50]},indent=2))
 index=max(range(100),key=lambda i:rows[i]['room_graph_hops'])
 env._set_state(rows[index]['start_xy']);env._set_goal_state(rows[index]['goals']['75']['xy'])
 im=Image.fromarray(env.render()).resize((448,448),Image.Resampling.NEAREST)
 draw=ImageDraw.Draw(im);points=[tuple(np.array(p)*2) for p in traces[index]];draw.line(points,fill=(20,120,220),width=3)
 g=np.array(rows[index]['goals']['75']['xy'])*2;draw.ellipse((g[0]-6,g[1]-6,g[0]+6,g[1]+6),fill=(0,180,0));panels.append((spec['id'],im))
 audit={'id':spec['id'],'rooms':nrooms,'edges':len(spec['edges']),'cycles':len(spec['edges'])-nrooms+1,'degrees':degrees,'diameter_hops':diameter,'reference_successes':100,'max_reference_steps':max(x['oracle_reference_steps'] for x in rows),'gate_dynamics_checks':gate_checks,'blocked_segment_checks':blocks,'random_validity_steps':random_steps,'render_mask_match':True,'noop_and_target_render_checks':100,'cases_sha256':hashlib.sha256((D/(spec['id']+'.json')).read_bytes()).hexdigest()}
 summary.append(audit);catalog.append(spec);print('MAP_PASSED',audit,flush=True)
# Bridge quantifies simulator-shift separately; differences are expected, not silently ignored.
spec=catalog[0];custom=TopologyEnv(spec);custom.reset(seed=5)
original=TwoRoomEnv(init_value={'wall.axis':1,'door.position':[49,49,49]});original.reset(seed=5)
diff=0;maxerr=0;free_mismatches=0;free_n=0
for i in range(1000):
 while True:
  p=rng.uniform(22,202,2).astype(np.float32)
  if custom.valid(p):break
 action=rng.uniform(-1,1,2).astype(np.float32)
 custom._set_state(p.copy());original._set_state(p.copy())
 custom.step(action);original.step(action)
 err=float(np.linalg.norm(custom.agent_position.numpy()-original.agent_position.numpy()));diff+=err>1e-5;maxerr=max(maxerr,err)
 if min(abs(float(p[0])-112),float(p.min())-14,210-float(p.max()))>25:
  free_n+=1;free_mismatches+=err>1e-5
assert free_mismatches==0
for e in [custom,original]:e._set_state(np.array([60,112],np.float32))
a=custom.render();b=original.render();pixel_diff=int(np.any(a!=b,axis=2).sum())
bridge={'n':1000,'one_step_different':diff,'max_position_difference':maxerr,'free_space_n':free_n,'free_space_differences':free_mismatches,'render_different_pixels':pixel_diff,'reason':'custom inflated rectangular obstacles and swept/axis-sliding collision differ from original center-in-door allowance; bridge is mandatory before topology claims'}
sheet=Image.new('RGB',(896,952),'white');draw=ImageDraw.Draw(sheet)
for i,(name,im) in enumerate(panels):
 x=(i%2)*448;y=(i//2)*476;draw.text((x+12,y+4),name,fill='black');sheet.paste(im,(x,y+24))
sheet.save(D/'topology_reference_overview.png')
payload={'passed':True,'created_unix':time.time(),'maps':catalog,'audit':summary,'bridge':bridge,'limits':['custom simulator, not an official benchmark','full observation, no memory test','new conservative collision semantics require bridge control','reference room-center route is feasible, not exact shortest path/SPL','400 development cases, not independent final test or learned-model results','original TwoRoom code and current GPU queue untouched']}
(D/'catalog.json').write_text(json.dumps(payload,indent=2));print('TOPOLOGY_CPU_AUDIT_PASSED',bridge,flush=True)
