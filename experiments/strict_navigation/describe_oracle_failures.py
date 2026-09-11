"""Describe the current target at failed oracle episodes without causal relabeling."""
import json,time
from pathlib import Path
from collections import Counter
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def read(p):return json.loads(p.read_text())
formal=read(R/'oracle_waypoint_formal_v1/results.json');assert formal['passed']
result={}
for mapid in formal['protocol']['maps']:
 result[mapid]={}
 for model in ['single','multi']:
  rows_out=[];counts=Counter()
  for batch in range(2):
   tag=f'oracle_formal_{model}_waypoint_{mapid}_b{batch}_v1';p=R/'runs'/tag
   rows=read(p/'cases_results.json');meta=read(p/'case_metadata.json');route=read(p/'oracle_waypoint_metadata.json')
   tr=np.load(p/'trace.npz');pos=tr['proprio'].reshape(len(tr['proprio']),50,-1)[:,:,:2]
   for i,(row,case,rr) in enumerate(zip(rows,meta['cases'],route['cases'])):
    if row['success']:counts['success']+=1;continue
    ix=rr['index'];last=len(rr['route'])-1
    typ=('same_room_final' if not row['different_room'] else 'final_goal_after_all_waypoints' if ix==last else 'first_door_entry' if ix==0 else 'door_exit' if ix%2 else 'later_door_entry')
    events=[e for e in route['events'] if e['episode']==row['episode']]
    start=next((e['call'] for e in events if e['after']==ix),0)
    # Distances only while this final unreached target was current.
    pre=np.array(case['start_xy']) if start==0 else pos[start-1,i]
    pts=np.vstack([pre,pos[start:row['steps'],i]])
    mind=float(np.linalg.norm(pts-np.array(rr['route'][ix]),axis=1).min())
    record={'episode':row['episode'],'room_graph_hops':case['room_graph_hops'],'failed_current_target_type':typ,
     'route_index':ix,'intermediate_reached':rr['intermediate_reached'],'intermediate_total':rr['intermediate_total'],
     'current_target_min_distance_px':mind,'current_target_active_since_call':start,
     'current_target_active_physical_steps':row['steps']-start,'endpoint_distance_to_final':row['endpoint_distance']}
    rows_out.append(record);counts[typ]+=1
  assert sum(counts.values())==100
  cross=[x for x in rows_out if x['failed_current_target_type']!='same_room_final']
  result[mapid][model]={'outcome_counts':dict(counts),'failed_cross_room_n':len(cross),
   'failed_current_intermediate_min_le16_gt8':sum(8<x['current_target_min_distance_px']<=16 for x in cross if x['route_index']<x['intermediate_total']),
   'failed_current_intermediate_min_le8_at_final_recorded_step':sum(x['current_target_min_distance_px']<=8 for x in cross if x['route_index']<x['intermediate_total']),
   'case_records':rows_out}
payload={'updated_unix':time.time(),'source':'oracle_waypoint_formal_v1/results.json','descriptive_only':True,
 'results':result,'limits':'Current unreached target type is not the causal failure reason.8px waypoint switching is stricter than16px final success; distances include last state with no following routing call. One training seed per condition; six fixed model/map groups.'}
(R/'reports/oracle_failure_stage_breakdown.json').write_text(json.dumps(payload,indent=2))
print(json.dumps({m:{v:{k:x[k] for k in ['outcome_counts','failed_cross_room_n','failed_current_intermediate_min_le16_gt8','failed_current_intermediate_min_le8_at_final_recorded_step']} for v,x in g.items()} for m,g in result.items()}),flush=True)
