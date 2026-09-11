"""Deadline-bounded experiment supervisor. Only manages its own child process groups."""
import os,sys,json,time,signal,subprocess,traceback
from pathlib import Path
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'round12h_20260909';PY=str(ROOT/'venv/bin/python')
DEADLINE=json.loads((R/'start.json').read_text())['deadline_unix']
state={'supervisor_pid':os.getpid(),'started_unix':time.time(),'deadline_unix':DEADLINE,'stage':'starting','active':None,'jobs':[]}
active=None
def save():
 p=R/'status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(state,indent=2));q.replace(p)
def report():
 try:subprocess.run([PY,str(R/'summarize_round.py')],timeout=45,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
 except Exception:pass
def stop(proc):
 if proc is None or proc.poll() is not None:return
 for sig,wait in [(signal.SIGINT,90),(signal.SIGTERM,20),(signal.SIGKILL,10)]:
  if proc.poll() is not None:return
  try:os.killpg(proc.pid,sig)
  except ProcessLookupError:return
  try:proc.wait(timeout=wait)
  except subprocess.TimeoutExpired:continue
def run(tag,args,quota,extra=None,reserve=180):
 global active
 remaining=DEADLINE-time.time()-reserve
 if remaining<30:
  state['jobs'].append({'tag':tag,'status':'skipped_budget','argv':args});save();return False
 quota=min(quota,remaining)
 log=R/'logs'/f'{tag}.log';log.parent.mkdir(exist_ok=True)
 env=os.environ.copy();env.update(extra or {});env['LEWM_RUN_TAG']=tag
 rec={'tag':tag,'argv':args,'started_unix':time.time(),'quota_seconds':quota,'log':str(log),'status':'running'}
 state['jobs'].append(rec)
 with log.open('w') as stream:
  active=subprocess.Popen(args,stdout=stream,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env,start_new_session=True)
  rec['pid']=active.pid;state['active']=rec;state['stage']=tag;save()
  timeout_at=time.time()+quota
  while active.poll() is None and time.time()<timeout_at:
   time.sleep(5)
  timed=active.poll() is None
  if timed:stop(active)
  rec.update({'finished_unix':time.time(),'returncode':active.poll(),'status':'timeout' if timed else 'complete' if active.returncode==0 else 'failed'})
  active=None;state['active']=None;save();report()
 return rec['status']=='complete'
def wait_primary():
 state['stage']='waiting_primary_training';save()
 p=R/'runs/r12_tw_s3072';limit=json.loads((p/'invocation.json').read_text())['start_unix']+11000
 while not (p/'training_summary.json').exists() and time.time()<min(limit,DEADLINE-180):
  if (p/'failure.json').exists():break
  try:os.kill(37503,0)
  except ProcessLookupError:break
  time.sleep(10)
 if not (p/'training_summary.json').exists():
  try:
   cmd=Path('/proc/37503/cmdline').read_bytes()
   if b'train_bounded.py' in cmd and b'r12_tw_s3072' in cmd:os.kill(37503,signal.SIGINT)
  except (FileNotFoundError,ProcessLookupError):pass
 state['jobs'].append({'tag':'r12_tw_s3072','status':'complete' if (p/'training_summary.json').exists() else 'incomplete','externally_started_pid':37503})
 save();report()
def dataset(env):
 if env=='tworoom':return 'tworoom'
 file=R/f'ready_datasets_{env}.json'
 if not file.exists():raise FileNotFoundError(str(file))
 files=[Path(p) for p in json.loads(file.read_text())['h5']]
 if not files:raise FileNotFoundError('No extracted HDF5 '+env)
 expected={'pusht':'pusht_expert_train','cube':'ogbench/cube_single_expert','reacher':'dmc/reacher_random'}[env]
 preferred=[p for p in files if p.stem==Path(expected).name]
 if not preferred and len(files)>1:raise RuntimeError('Ambiguous HDF5 files '+str(files))
 source=(preferred or files)[0]
 link=ROOT/'data/datasets'/(expected+'.h5');link.parent.mkdir(parents=True,exist_ok=True)
 if not link.exists():link.symlink_to(source)
 elif link.resolve()!=source.resolve():raise RuntimeError('Existing dataset link differs '+str(link))
 return expected
def evaluate(tag,env,policy,seed=42,overrides=None,quota=1500):
 try:name=dataset(env)
 except Exception as e:
  state['jobs'].append({'tag':tag,'status':'blocked_dataset','error':str(e)});save();return False
 nsteps=30 if env=='pusht' else 10
 args=[PY,str(R/'eval_record.py'),'--config-name',env,f'policy={policy}',f'seed={seed}',f'eval.dataset_name={name}',f'solver.n_steps={nsteps}',f'output.filename={R}/runs/{tag}_result.txt']+(overrides or [])
 return run(tag,args,quota,{'LEWM_VIDEO':'1'})
def weights(tag):return ROOT/'data/checkpoints'/tag/'weights_final.pt'
def train(tag,seed,seconds,extra=None,steps=0,env='tworoom'):
 seconds=min(seconds,DEADLINE-time.time()-1800)
 if seconds<600:
  state['jobs'].append({'tag':tag,'status':'skipped_budget'});save();return False
 data='tworoom' if env=='tworoom' else 'pusht'
 args=[PY,str(R/'train_bounded.py'),f'data={data}',f'seed={seed}',f'history_size={1 if env=="tworoom" else 3}','loss.sigreg.weight=0.1','trainer.max_epochs=10',f'subdir={tag}',f'output_model_name={tag}','num_workers=6','+trainer.enable_progress_bar=false']+(extra or [])
 return run(tag,args,seconds+120,{'LEWM_TRAIN_SECONDS':str(seconds),'LEWM_STOP_STEPS':str(steps)})
def diag(tag,path,quota=1200):
 if Path(path).exists():run('diag_'+tag,[PY,str(R/'diagnose_latent.py'),tag,str(path)],quota)
def main():
 wait_primary()
 # Core: all four environments, public weights and paired random baselines.
 for env in ['tworoom','pusht','reacher','cube']:
  public=ROOT/'data'/('hf_tworoom' if env=='tworoom' else 'hf_'+env)
  for seed in [42,43,44]:
   evaluate(f'public_{env}_s{seed}',env,str(public),seed)
   evaluate(f'random_{env}_s{seed}',env,'random',seed)
 # Primary from-scratch model, including longer targets.
 if weights('r12_tw_s3072').exists():
  for seed in [42,43,44]:
   evaluate(f'trained3072_tw_s{seed}','tworoom',str(weights('r12_tw_s3072')),seed)
   evaluate(f'trained3072_long_s{seed}','tworoom',str(weights('r12_tw_s3072')),seed,['eval.goal_offset_steps=75','eval.eval_budget=150'])
  diag('trained3072',weights('r12_tw_s3072'))
 # Paired, exploratory planner sensitivity; each setting is recorded, not called identical compute.
 variants={
  'long75':['eval.goal_offset_steps=75','eval.eval_budget=150'],
  'mid50':['eval.goal_offset_steps=50','eval.eval_budget=100'],
  'cem30':['solver.n_steps=30'],
  'long_h10':['eval.goal_offset_steps=75','eval.eval_budget=150','plan_config.horizon=10'],
  'long_replan1':['eval.goal_offset_steps=75','eval.eval_budget=150','plan_config.receding_horizon=1'],
  'long_samples64':['eval.goal_offset_steps=75','eval.eval_budget=150','solver.num_samples=64','solver.topk=8']
 }
 for name,overrides in variants.items():evaluate('sweep_'+name,'tworoom',str(ROOT/'data/hf_tworoom'),42,overrides)
 # Matched training-step regularization/dimension ablations. Same seed and nominal schedule.
 for tag,extra in [('r12_noreg',['loss.sigreg.weight=0.0']),('r12_lowreg',['loss.sigreg.weight=0.01']),('r12_dim64',['embed_dim=64','model.projector.input_dim=192'])]:
  if train(tag,3072,1000,extra,steps=3000):
   for seed in [42,43,44]:evaluate(f'{tag}_eval_s{seed}','tworoom',str(weights(tag)),seed)
   diag(tag,weights(tag),900)
 base3000=ROOT/'data/checkpoints/r12_tw_s3072/weights_step_3000.pt'
 if base3000.exists():
  for seed in [42,43,44]:evaluate(f'base3000_eval_s{seed}','tworoom',str(base3000),seed)
 # Replicate full navigation training if time permits, retaining evaluation time.
 for seed in [3073,3074]:
  tag=f'r12_tw_s{seed}'
  if DEADLINE-time.time()<10500:state['jobs'].append({'tag':tag,'status':'skipped_budget'});save();continue
  if train(tag,seed,9000):
   for es in [42,43,44]:evaluate(f'trained{seed}_tw_s{es}','tworoom',str(weights(tag)),es)
 # Retry evaluations deferred by asset downloads or transient failures before declaring finished.
 for env in ['pusht','reacher','cube']:
  for seed in [42,43,44]:
   for prefix,policy in [('public',str(ROOT/'data'/('hf_'+env))),('random','random')]:
    tag=f'{prefix}_{env}_s{seed}'
    if not (R/'runs'/tag/'metrics.json').exists():evaluate(tag,env,policy,seed,quota=1200)
 # Cross-task training transfer check; explicitly time-bounded, not a full PushT replication.
 if DEADLINE-time.time()>3600:
  try:
   dataset('pusht');tag='r12_pusht'
   if train(tag,3072,min(10800,DEADLINE-time.time()-1800),['data.dataset.name=pusht_expert_train.h5'],env='pusht'):
    evaluate('trained_pusht','pusht',str(weights(tag)),42)
  except Exception as e:state['jobs'].append({'tag':'r12_pusht','status':'blocked','error':repr(e)});save()
 state['stage']='complete';state['finished_unix']=time.time();save();report()
if __name__=='__main__':
 try:
  if '--self-test' in sys.argv:
   assert run('supervisor_selftest',[PY,'-c','print("supervisor OK")'],15)
   assert not run('supervisor_timeout_test',[PY,'-c','import time; time.sleep(60)'],1)
   assert state['jobs'][-1]['status']=='timeout'
   print('SELF_TEST_OK')
  else:main()
 except BaseException as e:
  state['stage']='interrupted' if isinstance(e,KeyboardInterrupt) else 'failed';state['error']=traceback.format_exc();save()
 finally:
  stop(active);state['active']=None;save();report()
