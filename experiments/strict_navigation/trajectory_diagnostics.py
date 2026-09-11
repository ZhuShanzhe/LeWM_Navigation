"""Geometry-assisted diagnosis of completed logged trajectories; never used by controllers."""
import json,re
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
state=json.loads((R/'status.json').read_text());groups={}
for job in state['jobs']:
 tag=job['tag']
 if job['status']!='complete' or not re.match(r'strict\d+_(test|validation)_',tag):continue
 out=R/'runs'/tag
 if not (out/'trace.npz').exists():continue
 rows=json.loads((out/'cases_results.json').read_text());meta=json.loads((out/'case_metadata.json').read_text())
 trace=np.load(out/'trace.npz');group=re.sub(r'_b\d+$','',tag);record=groups.setdefault(group,[])
 for i,row in enumerate(rows):
  n=row['steps'];xy=trace['proprio'][:n,i].reshape(n,-1)[:,:2];a=trace['actions'][:n,i].reshape(n,-1)[:,:2]
  start=np.asarray(meta['cases'][i]['start_xy']);goal=np.asarray(meta['cases'][i]['goals'][str(meta['goal_offset'])]['xy'])
  full=np.vstack([start,xy]);moves=np.linalg.norm(np.diff(full,axis=0),axis=1)
  crossed=np.any((xy[:,0]-112)*(goal[0]-112)>0) if row['across_wall'] else False
  # Proxy region, not contact-event instrumentation.
  wall_region=(np.abs(xy[:,0]-112)<=14)&(np.abs(xy[:,1]-49)>=7)
  wall_stall=(moves<.5)&wall_region
  clipped=np.abs(a)>1+1e-6;physical=np.clip(a,-1,1);norm=np.linalg.norm(physical,axis=1)
  dot=(physical[1:]*physical[:-1]).sum(1);valid=(norm[1:]>1e-6)&(norm[:-1]>1e-6)
  reversal=(dot/np.maximum(norm[1:]*norm[:-1],1e-10)<-.5)&valid
  if row['success']:category='success'
  elif row['across_wall'] and not crossed:category='cross_wall_failed_before_crossing'
  elif row['across_wall']:category='cross_wall_failed_after_crossing'
  else:category='same_side_failure'
  record.append({'episode':row['episode'],'start_step':row['start_step'],'success':row['success'],'initial_success':row['initial_success'],'category':category,'steps':n,'near_wall_stalled_steps':int(wall_stall.sum()),'minimum_goal_distance':float(np.linalg.norm(xy-goal,axis=1).min()),'executed_action_components_out_of_bounds':int(clipped.sum()),'executed_action_components':int(a.size),'action_direction_reversals_gt120deg':int(reversal.sum()),'wall_crossings':int(np.sum((full[:-1,0]-112)*(full[1:,0]-112)<0))})
payload={}
for group,rows in groups.items():
 counts={k:sum(r['category']==k for r in rows) for k in ['success','cross_wall_failed_before_crossing','cross_wall_failed_after_crossing','same_side_failure']}
 payload[group]={'n':len(rows),'failure_categories':counts,
 'executed_action_component_clipping_fraction':sum(r['executed_action_components_out_of_bounds'] for r in rows)/max(1,sum(r['executed_action_components'] for r in rows)),
 'mean_near_wall_stalled_steps_on_failures':float(np.mean([r['near_wall_stalled_steps'] for r in rows if not r['success']])) if any(not r['success'] for r in rows) else None,
 'mean_action_reversals':float(np.mean([r['action_direction_reversals_gt120deg'] for r in rows])),
 'rows':rows}
result={'groups':payload,'limitations':['Wall proximity stalls are a geometric proxy, not instrumented collision counts.','Action clipping is measured on submitted physical actions, not candidate pools.','Crossing classification only applies to the original vertical-wall map.','Direction reversals are descriptive; they do not alone diagnose missing memory.']}
(R/'reports/trajectory_diagnostics.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:{x:y for x,y in v.items() if x!='rows'} for k,v in payload.items()},indent=2))
