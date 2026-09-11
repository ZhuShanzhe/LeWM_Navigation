"""Run official train.py with reproducible seeds and local CSV logging."""
import json,os,runpy,sys,time
from pathlib import Path
import torch
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
root=Path('/root/autodl-tmp/lewm_research'); repo=root/'le-wm'
sys.path.insert(0,str(repo)); os.chdir(repo)
seed=next((int(a.split('=',1)[1]) for a in sys.argv[1:] if a.startswith('seed=')),3072)
pl.seed_everything(seed,workers=True); torch.set_num_threads(8)
tag=os.environ.get('LEWM_RUN_TAG','train_pilot')
folder=root/'runs'/tag; folder.mkdir(parents=True,exist_ok=True)
(folder/'invocation.json').write_text(json.dumps({'argv':sys.argv,'seed':seed},indent=2))
OriginalTrainer=pl.Trainer
class LoggedTrainer(OriginalTrainer):
 def __init__(self,*args,**kwargs):
  if kwargs.get('logger') is None: kwargs['logger']=CSVLogger(str(folder),name='csv',version='')
  kwargs['default_root_dir']=str(folder)
  kwargs['log_every_n_steps']=10
  super().__init__(*args,**kwargs)
 def fit(self,*args,**kwargs):
  start=time.time()
  result=super().fit(*args,**kwargs)
  self.save_checkpoint(str(folder/'last.ckpt'))
  metrics={str(k):float(v.detach().cpu()) if isinstance(v,torch.Tensor) and v.numel()==1 else str(v) for k,v in self.callback_metrics.items()}
  (folder/'training_summary.json').write_text(json.dumps({'global_step':self.global_step,'current_epoch':self.current_epoch,'elapsed_seconds':time.time()-start,'metrics':metrics,'gpu':torch.cuda.get_device_name()},indent=2))
  return result
pl.Trainer=LoggedTrainer
runpy.run_path(str(repo/'train.py'),run_name='__main__')
