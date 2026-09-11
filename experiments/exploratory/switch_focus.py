import os,sys,json,signal,time,subprocess
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909');p=R/'status.json';s=json.loads(p.read_text())
assert s['supervisor_pid']==38948 and s['active']['tag']=='r12_tw_s3073' and s['active']['pid']==149240,s
assert b'supervisor.py' in Path('/proc/38948/cmdline').read_bytes()
assert b'train_bounded.py' in Path('/proc/149240/cmdline').read_bytes()
archive=R/'status_before_focus.json'
assert not archive.exists(),'Transition already performed'
archive.write_text(json.dumps(s,indent=2))
# Only terminate the old queue controller, not its separately sessioned GPU child.
os.kill(38948,signal.SIGTERM);time.sleep(1)
assert b'r12_tw_s3073' in Path('/proc/149240/cmdline').read_bytes(),'Current training unexpectedly exited'
seconds=max(1,int(s['deadline_unix']-time.time()))
with (R/'focused_supervisor.log').open('w') as f:
 proc=subprocess.Popen(['timeout','--signal=INT','--kill-after=120s',str(seconds),sys.executable,str(R/'focused_supervisor.py')],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
(R/'focus_transition.json').write_text(json.dumps({'at':time.time(),'old_supervisor':38948,'preserved_training':149240,'new_wrapper':proc.pid,'deadline_unix':s['deadline_unix'],'reason':'User prioritizes diagnostic validity over full reproduction'},indent=2))
print('FOCUSED_WRAPPER_PID',proc.pid)
