"""Retry this round's downloads, tracking replacement processes and fixed deadline."""
import os,time,json,subprocess,signal
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909')
deadline=json.loads((R/'start.json').read_text())['deadline_unix']-3600
attempts={};children={}
def alive(pid):
 try:return b'download_all.py' in Path(f'/proc/{pid}/cmdline').read_bytes()
 except FileNotFoundError:return False
def ready(e):
 try:return bool(json.loads((R/f'ready_datasets_{e}.json').read_text()).get('h5'))
 except:return False
while time.time()<deadline:
 for e,pid in [('pusht',37863),('reacher',39774),('cube',39774)]:
  if ready(e) or alive(pid) or attempts.get(e,0)>=2:continue
  old=children.get(e)
  if old and old.poll() is None:continue
  env=os.environ.copy();env['LEWM_DOWNLOAD_ENVS']=e
  log=(R/f'asset_retry_{e}_{attempts.get(e,0)}.log').open('w')
  cmd=f'source /etc/network_turbo >/dev/null 2>&1; source {R}/env.sh; exec python {R}/download_all.py'
  children[e]=subprocess.Popen(['bash','-lc',cmd],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close()
  attempts[e]=attempts.get(e,0)+1
  (R/'asset_retry_status.json').write_text(json.dumps({'attempts':attempts,'pids':{k:v.pid for k,v in children.items()}}))
 if all(ready(e) for e in ['pusht','reacher','cube']):break
 time.sleep(30)
for e,p in children.items():
 if p.poll() is None:
  os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=20)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL)
print('ASSET_WATCH_COMPLETE')
