"""Paired validation comparisons specifically for continuity and action bounds."""
import json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def rows(name):
 out={}
 for b in [0,1]:
  p=R/'runs'/f'strict3072_validation_{name}_b{b}'/'cases_results.json'
  if not p.exists():return None
  for row in json.loads(p.read_text()):out[(row['episode'],row['start_step'])]=row
 return out
rng=np.random.default_rng(9102045);result=[]
for a,b in [('long_feedback_extra','long_feedback_extra_warm'),('long_h10_budget','long_h10_budget_warm'),('long_feedback_budget','long_feedback_budget_warm'),('long_base','long_base_bounded'),('long_feedback_extra','long_feedback_extra_bounded'),('long_feedback_extra_warm','long_feedback_extra_warm_bounded')]:
 ra,rb=rows(a),rows(b)
 if ra is None or rb is None:continue
 assert ra.keys()==rb.keys() and len(ra)==100
 d=np.array([int(rb[k]['success'])-int(ra[k]['success']) for k in sorted(ra)])
 boots=d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1)*100
 result.append({'baseline':a,'variant':b,'n':len(d),'delta_pp':100*float(d.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),'wins':int((d==1).sum()),'losses':int((d==-1).sum()),'scope':'one model, development validation; not independent training-seed confidence'})
(R/'reports/continuity_controls.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
