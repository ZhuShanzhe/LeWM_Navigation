"""Single serial job queue. Locks own folder; finite jobs; stops at review gate."""
import os,sys,json,time,subprocess,signal,fcntl,traceback
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
lock=(R/'queue.lock').open('w')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
statepath=R/'status.json'
state=json.loads(statepath.read_text()) if statepath.exists() else {'jobs':[]}
state.update(supervisor_pid=os.getpid(),stage='starting',active=None,updated_unix=time.time())
def save():
 state['updated_unix']=time.time();p=statepath.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2));p.replace(statepath)
active=None
def stop():
 if active is None or active.poll() is not None:return
 for sig,wait in [(signal.SIGINT,90),(signal.SIGTERM,20),(signal.SIGKILL,5)]:
  if active.poll() is not None:break
  try:os.killpg(active.pid,sig);active.wait(timeout=wait)
  except (ProcessLookupError,subprocess.TimeoutExpired):pass
try:
 save()
 while True:
  queue=json.loads((R/'jobs.json').read_text())
  seen={x['tag'] for x in state['jobs']}
  pending=[j for j in queue if j['tag'] not in seen]
  if not pending:state['stage']='awaiting_next_stage_review';save();break
  j=pending[0];env=os.environ.copy();env.update(j.get('env',{}));env['LEWM_RUN_TAG']=j['tag']
  log=R/'logs'/(j['tag']+'.log')
  rec={**j,'started_unix':time.time(),'status':'running','log':str(log)}
  state['jobs'].append(rec);state['active']=rec;state['stage']=j['tag'];save()
  with log.open('x') as f:
   active=subprocess.Popen(j['argv'],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
   rec['pid']=active.pid;save()
   try:active.wait(timeout=j.get('watchdog_seconds',43200));timed=False
   except subprocess.TimeoutExpired:timed=True;stop()
  code=active.returncode;active=None;ok=code==0
  if j.get('training'):
   summary=R/'runs'/j['tag']/'training_summary.json'
   ok=ok and summary.exists() and json.loads(summary.read_text()).get('full_epoch_budget_completed',False)
  rec.update(status='complete' if ok else 'watchdog' if timed else 'failed',returncode=code,finished_unix=time.time())
  state['active']=None;save()
  if not ok:state['stage']='needs_failure_review';save();break
except BaseException:
 state.update(stage='needs_failure_review',error=traceback.format_exc());save();raise
finally:
 stop();state['active']=None;save()
