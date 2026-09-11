"""Training-only visual trajectory retrieval. No test/map coordinates used."""
import os,json,hashlib,time
from pathlib import Path
import numpy as np,torch,h5py,hdf5plugin
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build():
 source=R/'heads/temporal3072';out=R/'priors/retrieval3072';out.mkdir(parents=True,exist_ok=True)
 target=out/'library.pt'
 if target.exists():return
 data=torch.load(source/'encoded_pairs.pt',map_location='cpu',weights_only=False)
 with np.load(source/'pair_indices.npz') as p:
  pairs=p['train'];unique=p['unique_rows']
  forward=np.sort(pairs[:,:2],axis=1);keep=pairs[:,2]>=25
  selected=np.c_[forward[keep],pairs[keep,3]]
  selected=np.unique(selected,axis=0)
 manifest=json.loads((R/'splits/manifest.json').read_text())
 assert set(selected[:,2])<=set(manifest['episodes']['train'])
 assert not set(selected[:,2])&set(manifest['episodes']['test'])
 with h5py.File(R.parent/'data/tworoom.h5','r') as f:
  actions=f['action'][:];ends=f['ep_offset'][:]+f['ep_len'][:]
  assert np.all(selected[:,0]+25<=ends[selected[:,2]])
  chunks=actions[selected[:,0,None]+np.arange(25)[None,:]]
 stats=json.loads((R/'splits/normalization.json').read_text())['action']
 normalized=(chunks-np.array(stats['mean']))/np.array(stats['std'])
 ix=np.searchsorted(unique,selected[:,:2])
 assert np.array_equal(unique[ix],selected[:,:2])
 library={'start':data['latents'][ix[:,0]],'goal':data['latents'][ix[:,1]],
  'actions':torch.from_numpy(normalized.astype(np.float32)).reshape(-1,5,10),
  'rows':torch.from_numpy(selected),'checkpoint_sha256':data['checkpoint_sha256'],
  'split_sha256':sha(R/'splits/manifest.json'),'physical_horizon':25}
 torch.save(library,target)
 (out/'audit.json').write_text(json.dumps({'library_entries':len(selected),
  'train_episodes':len(set(selected[:,2])),'test_overlap':0,'validation_overlap':0,
  'source_pair_indices_sha256':sha(source/'pair_indices.npz'),'library_sha256':sha(target),
  'checkpoint_sha256':data['checkpoint_sha256'],'split_sha256':library['split_sha256'],
  'forward_only':True,'horizon':25,'filter':'temporal separation >=25; unique start-goal-episode tuples',
  'selection':'raw latent squared start distance plus goal distance, equal weights',
  'limitation':'logged behavior prior; future anchor may be farther than the 25-step action chunk; not shortest-path or expert supervision',
  'physical_action_max_abs':float(np.max(np.abs(chunks)))},indent=2))
 print('RETRIEVAL_LIBRARY',len(selected),flush=True)

def install(swm):
 original=swm.solver.CEMSolver.solve
 @torch.inference_mode()
 def solve(self,info_dict,init_action=None):
  from pathlib import Path
  library_path=os.environ['LEWM_RETRIEVAL_LIBRARY'];mode=os.environ['LEWM_RETRIEVAL_MODE']
  assert mode in ['only','rerank','init']
  assert self.horizon==5 and self.action_dim==10,'retrieval protocol requires 25 physical steps'
  device=next(self.model.parameters()).device
  if not hasattr(self,'_retrieval_library'):
   lib=torch.load(library_path,map_location='cpu',weights_only=False)
   # model path is supplied by existing eval CLI, and verified before any retrieval.
   import sys
   modelpath=next(a.split('=',1)[1] for a in sys.argv if a.startswith('policy='))
   assert sha(modelpath)==lib['checkpoint_sha256']
   assert sha(R/'splits/manifest.json')==lib['split_sha256']
   self._retrieval_library={k:v.to(device) if torch.is_tensor(v) else v for k,v in lib.items()}
   self._retrieval_log={'mode':mode,'library':library_path,'library_sha256':sha(library_path),
    'entries':len(lib['start']),'calls':[],'no_geometry_input':True,'train_only':True}
  lib=self._retrieval_library;beg=time.perf_counter()
  start=self.model.encode({'pixels':info_dict['pixels'].to(device)})['emb'][:,-1]
  goal=self.model.encode({'pixels':info_dict['goal'].to(device)})['emb'][:,-1]
  # Squared Euclidean pair matching; constants depending only on query omitted.
  score=(lib['start'].square().sum(1)+lib['goal'].square().sum(1))[None,:]-2*(start@lib['start'].T+goal@lib['goal'].T)
  k=self.num_samples if mode=='rerank' else 1
  selected=score.topk(k,largest=False,dim=1).indices
  proposed=lib['actions'][selected]
  if device.type=='cuda':torch.cuda.synchronize()
  elapsed=time.perf_counter()-beg
  self._retrieval_log['calls'].append({'queries':len(start),'pair_distances':len(start)*len(lib['start']),
   'retrieval_seconds':elapsed,'candidates_returned':len(start)*k})
  if mode=='init':
   result=original(self,info_dict,init_action=proposed[:,0].cpu())
  elif mode=='only':
   result={'actions':proposed[:,0].cpu(),'costs':[],'mean':[],'var':[]}
  else:
   results=[];costs=[]
   for i in range(len(start)):
    expanded={}
    for key,value in info_dict.items():
     if torch.is_tensor(value):
      v=value[i:i+1].to(device);expanded[key]=v[:,None].expand(1,k,*v.shape[1:])
     elif isinstance(value,np.ndarray):expanded[key]=np.repeat(value[i:i+1,None],k,axis=1)
    candidate=proposed[i:i+1].clone();cost=self.model.get_cost(expanded,candidate)
    idx=int(cost[0].argmin());results.append(candidate[0,idx].cpu());costs.append(float(cost[0,idx]))
   result={'actions':torch.stack(results),'costs':costs,'mean':[],'var':[]}
  out=R/'runs'/os.environ['LEWM_RUN_TAG']
  (out/'retrieval_metadata.json').write_text(json.dumps(self._retrieval_log,indent=2))
  return result
 swm.solver.CEMSolver.solve=solve
if __name__=='__main__':build()
