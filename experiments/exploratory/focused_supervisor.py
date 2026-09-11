"""Focused navigation diagnostics; replaces pending breadth-first queue, preserves current training."""
import os,sys,time,json,signal,subprocess,importlib.util,traceback
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909')
spec=importlib.util.spec_from_file_location('queue_helpers',R/'supervisor.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def report():
 for script in ['summarize_round.py','summarize_focused.py']:
  try:subprocess.run([m.PY,str(R/script)],timeout=45,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  except Exception:pass
m.report=report
def adopt_second():
 old=json.loads((R/'status_before_focus.json').read_text());m.state=old
 m.state.update(supervisor_pid=os.getpid(),queue_revision='focused-navigation-v1',stage='waiting_second_training',plan_change='User prioritizes reliable limitation diagnosis; cancel third seed and PushT training.')
 m.state['jobs'] += [{'tag':t,'status':'cancelled_user_priority'} for t in ['r12_tw_s3074','r12_pusht']]
 rec=m.state['active'];pid=rec['pid'];m.save()
 summary=R/'runs/r12_tw_s3073/training_summary.json';limit=min(rec['started_unix']+rec['quota_seconds'],m.DEADLINE-180)
 while time.time()<limit:
  proc=Path(f'/proc/{pid}/cmdline')
  if not proc.exists() or not proc.read_bytes():break
  if (R/'runs/r12_tw_s3073/failure.json').exists():break
  time.sleep(10)
 if not summary.exists():
  try:
   cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
   if b'train_bounded.py' in cmd and b'r12_tw_s3073' in cmd:
    os.killpg(pid,signal.SIGINT);time.sleep(15)
    if Path(f'/proc/{pid}/cmdline').exists() and Path(f'/proc/{pid}/cmdline').read_bytes():os.killpg(pid,signal.SIGTERM)
  except ProcessLookupError:pass
 rec.update(status='complete' if summary.exists() else 'incomplete',finished_unix=time.time(),adopted_by_focused_queue=True)
 for job in m.state['jobs']:
  if job.get('pid')==pid:job.update(rec)
 m.state['active']=None;m.save();report()
def case_eval(label,path,variant,seed,extra,noop=False):
 tag=f'focus_{label}_{variant}_s{seed}'
 if (R/'runs'/tag/'metrics.json').exists():return
 args=[m.PY,str(R/'eval_focused.py'),'--config-name','tworoom',f'policy={path}',f'seed={seed}','solver.n_steps=10','eval.num_eval=50',f'output.filename={R}/runs/{tag}_result.txt']+extra
 m.run(tag,args,1200,{'LEWM_CASE_FILE':str(R/'focused'/f'cases_s{seed}.json'),'LEWM_NOOP':'1' if noop else '0','LEWM_VIDEO':'1'})
def main():
 adopt_second()
 models=[('public',str(m.ROOT/'data/hf_tworoom')),('trained3072',str(m.weights('r12_tw_s3072'))),('trained3073',str(m.weights('r12_tw_s3073')))]
 models=[(k,p) for k,p in models if Path(p).exists()]
 # Preserve comparable official protocol result for second independently trained model.
 if m.weights('r12_tw_s3073').exists():
  for seed in [42,43,44]:m.evaluate(f'trained3073_tw_s{seed}','tworoom',str(m.weights('r12_tw_s3073')),seed)
  m.diag('trained3073',m.weights('r12_tw_s3073'),600)
 # Establish static and random baselines on the exact same cases.
 for seed in [42,43,44]:
  for v,g,b in [('short',25,50),('long',75,150)]:
   for label,noop in [('noop',True),('random',False)]:
    case_eval(label,'random',v,seed,[f'eval.goal_offset_steps={g}',f'eval.eval_budget={b}'],noop)
 # Core paired comparison. Short150 controls the execution budget, not goal identity/distance.
 variants={'short':['eval.goal_offset_steps=25','eval.eval_budget=50'],
 'short150':['eval.goal_offset_steps=25','eval.eval_budget=150'],
 'long':['eval.goal_offset_steps=75','eval.eval_budget=150']}
 for v,over in variants.items():
  for label,path in models:
   for seed in [42,43,44]:case_eval(label,path,v,seed,over)
 # Four environments public results remain context, not the central research outcome.
 for seed in [42,43,44]:
  for label,path in [('public',str(m.ROOT/'data/hf_cube')),('random','random')]:
   m.evaluate(f'{label}_cube_s{seed}','cube',path,seed,quota=900)
 # Mechanism tests: same goals/budget. Rough-compute controls are explicitly not exact.
 long=['eval.goal_offset_steps=75','eval.eval_budget=150']
 variants={'long_cem30':['solver.n_steps=30'],
 'long_h10_n150':['plan_config.horizon=10','solver.num_samples=150','solver.topk=15'],
 'long_replan1_iter2':['plan_config.receding_horizon=1','solver.n_steps=2'],
 'long_replan1_iter10':['plan_config.receding_horizon=1']}
 for v,over in variants.items():
  for label,path in models:
   for seed in [42,43,44]:
    if m.DEADLINE-time.time()<900:
     m.state['jobs'].append({'tag':f'focus_{label}_{v}_s{seed}','status':'skipped_budget'});m.save();continue
    case_eval(label,path,v,seed,long+over)
 m.state.update(stage='complete',finished_unix=time.time(),active=None);m.save();report()
if __name__=='__main__':
 try:main()
 except BaseException:
  m.state.update(stage='failed',error=traceback.format_exc());m.save()
 finally:m.stop(m.active);m.state['active']=None;m.save();report()
