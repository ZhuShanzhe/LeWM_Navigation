"""CPU closed-loop route switching at8/16px, unlike nominal exact-waypoint feasibility."""
from pathlib import Path
import json,time,hashlib
import numpy as np,torch
from oracle_waypoint_router import OracleWaypointRouter
from topology_env import TopologyEnv
torch.set_num_threads(1)
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
results={};checks={}
for mapid in ['four_room_chain','four_room_cycle','six_room_branch']:
 d=json.loads((R/'topology_dev_v2'/f'{mapid}.json').read_text());e=TopologyEnv(d['topology_spec']);e.reset(seed=9105121)
 routes=OracleWaypointRouter(d['topology_spec'],d['cases']).routes;results[mapid]={}
 for threshold in [8.,16.]:
  records=[]
  for c,route in zip(d['cases'],routes):
   e._set_state(np.array(c['start_xy'],np.float32));e._set_goal_state(np.array(c['goals']['75']['xy'],np.float32));ix=0;success=False;switches=0;blocked_lines=0
   for t in range(150):
    pos=e.agent_position.numpy()
    while ix<len(route)-1 and np.linalg.norm(pos-route[ix])<=threshold:
     ix+=1;switches+=1
     if not e.clear(pos,route[ix]):blocked_lines+=1
    _,_,success,_,_=e.step(np.clip((route[ix]-pos)/5,-1,1).astype(np.float32))
    assert e.valid(e.agent_position.numpy())
    if success:break
   records.append({'episode':c['episode'],'success':bool(success),'steps':t+1,'switches':switches,'blocked_next_line_at_switch':blocked_lines})
  results[mapid][str(int(threshold))]={'n':len(records),'successes':sum(x['success'] for x in records),'max_steps':max(x['steps'] for x in records),'blocked_next_lines':sum(x['blocked_next_line_at_switch'] for x in records),'cases':records}
 e.close()
passed=all(v['successes']==100 for g in results.values() for v in g.values())
out={'passed':passed,'updated_unix':time.time(),'results':results,'source_sha256':hashlib.sha256((R/'oracle_waypoint_router.py').read_bytes()).hexdigest(),
 'limits':'True-state direct low-level controller and actual8/16px switching rule, not learned policy. This complements earlier nominal exact-waypoint feasibility which did not execute switching tolerance.'}
(R/'reports/oracle_threshold_cpu_audit.json').write_text(json.dumps(out,indent=2))
print('THRESHOLD_CPU',passed,json.dumps({m:{k:{v:x[v] for v in ['successes','max_steps','blocked_next_lines']} for k,x in g.items()} for m,g in results.items()}),flush=True)
