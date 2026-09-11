"""Paper-protocol runner: unchanged objective/model, bounded execution, explicit artifacts."""
import os,sys,json,time,runpy,traceback
from pathlib import Path
import torch,lightning as pl
from lightning.pytorch.callbacks import Callback
from omegaconf import OmegaConf
import stable_worldmodel as swm
root=Path('/root/autodl-tmp/lewm_research');work=root/'round12h_20260909'
repo=root/'le-wm';sys.path.insert(0,str(repo));os.chdir(repo)
tag=os.environ['LEWM_RUN_TAG'];folder=work/'runs'/tag;folder.mkdir(parents=True,exist_ok=True)
seed=next((int(x.split('=',1)[1]) for x in sys.argv if x.startswith('seed=')),3072)
stop_steps=int(os.environ.get('LEWM_STOP_STEPS','0'));quota=float(os.environ.get('LEWM_TRAIN_SECONDS','10800'))
start=time.time();pl.seed_everything(seed,workers=True);torch.set_num_threads(8)
def atomic(name,value):
 p=folder/name;q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(value,indent=2));q.replace(p)
atomic('invocation.json',{'argv':sys.argv,'seed':seed,'quota_seconds':quota,'stop_steps':stop_steps,'start_unix':start})
def scalars(t):
 return {str(k):float(v.detach().cpu()) for k,v in t.callback_metrics.items() if isinstance(v,torch.Tensor) and v.numel()==1}
def save_model(trainer,name):
 cfg=OmegaConf.load(root/'data/checkpoints'/tag/'config.yaml')
 swm.wm.utils.save_pretrained(trainer.lightning_module.model,run_name=tag,config=cfg.model,filename=name)
class Audit(Callback):
 def on_train_batch_end(self,trainer,pl_module,outputs,batch,batch_idx):
  n=trainer.global_step
  if n%100==0:
   atomic('progress.json',{'step':n,'epoch':trainer.current_epoch,'batches_per_epoch':trainer.num_training_batches,'elapsed_seconds':time.time()-start,'metrics':scalars(trainer),'updated_unix':time.time()})
  if n in [3000,10000,30000]:save_model(trainer,f'weights_step_{n}.pt')
  if (stop_steps and n>=stop_steps) or time.time()-start>=quota-120:trainer.should_stop=True
Original=pl.Trainer
class Trainer(Original):
 def __init__(self,*a,**kw):
  kw['callbacks']=[*kw.get('callbacks',[]),Audit()]
  kw['log_every_n_steps']=50;kw['default_root_dir']=str(folder)
  super().__init__(*a,**kw)
 def fit(self,*a,**kw):
  try:
   answer=super().fit(*a,**kw)
   self.save_checkpoint(str(folder/'last.ckpt'));save_model(self,'weights_final.pt')
   atomic('training_summary.json',{'global_step':self.global_step,'epoch_counter':self.current_epoch,'max_epochs':self.max_epochs,'batches_per_epoch':self.num_training_batches,'epoch_equivalent':self.global_step/self.num_training_batches,'elapsed_seconds':time.time()-start,'metrics':scalars(self),'completed_unix':time.time(),'full_epoch_budget_completed':self.global_step>=self.max_epochs*self.num_training_batches,'weights':str(root/'data/checkpoints'/tag/'weights_final.pt')})
   return answer
  except BaseException as exc:
   atomic('failure.json',{'exception':repr(exc),'traceback':traceback.format_exc(),'step':self.global_step,'time':time.time()})
   raise
pl.Trainer=Trainer
runpy.run_path(str(repo/'train.py'),run_name='__main__')
