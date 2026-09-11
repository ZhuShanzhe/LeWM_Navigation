import json
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
j=json.loads((R/'jobs.json').read_text());tag='launch_geometry_collections_cpu'
if not any(x['tag']==tag for x in j):
 idx=next(i for i,x in enumerate(j) if x['tag']=='strict3072_online_long_h10_budget')+1
 j.insert(idx,{'tag':tag,'watchdog_seconds':120,'argv':[str(R.parent/'venv/bin/python'),str(R/'launch_geometry_collections.py')],'env':{},'note':'Launch only; collectors complete separately in collections/*/summary.json. CPU workers start after online latency controls.'})
 tmp=R/'jobs.tmp';tmp.write_text(json.dumps(j,indent=2));tmp.replace(R/'jobs.json')
print('queue',len(j))
