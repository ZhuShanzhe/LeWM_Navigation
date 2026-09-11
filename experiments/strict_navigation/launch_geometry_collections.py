"""Launch two finite CPU-only collectors after online latency baselines; no GPU processes."""
import os,json,subprocess,time
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
aud=json.loads((R/'collection_smoke_v2/audit.json').read_text())
assert all(x['passed'] for x in aud.values())
out=R/'collections';out.mkdir(exist_ok=True)
state=out/'launcher.json';assert not state.exists(),'Collection launch already recorded; inspect before retry'
records=[]
for variant in ['single','multi']:
 log=out/(variant+'.log')
 env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
 env.pop('LEWM_COLLECTION_SMOKE',None)
 with log.open('x') as f:
  proc=subprocess.Popen(['timeout','10800',str(R.parent/'venv/bin/python'),str(R/'collect_geometry_control.py'),variant],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
 records.append({'variant':variant,'pid':proc.pid,'log':str(log),'started_unix':time.time(),'timeout_seconds':10800,'device':'cpu','status':'launched_not_completed'})
 state.write_text(json.dumps(records,indent=2))
print('CPU_COLLECTIONS_LAUNCHED',json.dumps(records))
