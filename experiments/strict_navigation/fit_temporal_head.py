"""Strict TRM-inspired head and shuffled-label control, selected solely by heldout pair loss."""
import os,sys,json,time,hashlib,copy
from pathlib import Path
import numpy as np,h5py,hdf5plugin,torch
import stable_worldmodel as swm
from scipy.stats import spearmanr
from reachability_model import TemporalHead
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');ROOT=R.parent
sys.path.insert(0,str(ROOT/'le-wm'))
tag,modelpath=sys.argv[1:3];out=R/'heads'/tag;out.mkdir(parents=True,exist_ok=True)
device=os.environ.get('LEWM_HEAD_DEVICE','cuda');seed=int(os.environ.get('LEWM_HEAD_SEED','9102043'))
ntrain=int(os.environ.get('LEWM_HEAD_NTRAIN','100000'));nval=int(os.environ.get('LEWM_HEAD_NVAL','10000'));epochs=int(os.environ.get('LEWM_HEAD_EPOCHS','20'))
manifest=json.loads((R/'splits/manifest.json').read_text())
torch.set_num_threads(4);torch.manual_seed(seed)
if device=='cuda':torch.cuda.manual_seed_all(seed)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checkpoint_sha=sha(modelpath);cache=out/'encoded_pairs.pt';start=time.time()
def pairs(epids,lens,offset,n,rs):
 bins=[(1,15),(16,30),(31,50),(51,75),(76,int(lens[epids].max())-1)]
 rows=[];counts=[]
 for k,(lo,hi) in enumerate(bins):
  eligible=np.array([e for e in epids if lens[e]>lo],dtype=np.int64)
  if not len(eligible):raise ValueError(('Empty temporal bin',lo,hi))
  count=n//len(bins)+(k<n%len(bins));counts.append(count)
  for ep in rs.choice(eligible,count,replace=True):
   delta=int(rs.integers(lo,min(hi,int(lens[ep])-1)+1));t=int(rs.integers(0,int(lens[ep])-delta))
   a=int(offset[ep])+t;b=a+delta
   if rs.random()<.5:a,b=b,a
   rows.append((a,b,delta,int(ep),k))
 return np.array(rows,dtype=np.int64),counts
if not cache.exists():
 with h5py.File(ROOT/'data/tworoom.h5','r') as f:
  lens=f['ep_len'][:];offset=f['ep_offset'][:]
  train,tc=pairs(manifest['episodes']['train'],lens,offset,ntrain,np.random.default_rng(seed))
  val,vc=pairs(manifest['episodes']['validation'],lens,offset,nval,np.random.default_rng(seed+1))
  assert not set(train[:,3])&set(val[:,3])
  assert not (set(train[:,3])|set(val[:,3]))&set(manifest['episodes']['test'])
  wanted=np.unique(np.concatenate([train[:,:2].ravel(),val[:,:2].ravel()]))
  model=swm.wm.utils.load_pretrained(modelpath).to(device).eval().requires_grad_(False)
  mean=torch.tensor([.485,.456,.406],device=device)[None,:,None,None];std=torch.tensor([.229,.224,.225],device=device)[None,:,None,None]
  zs=[]
  batch=128 if device=='cuda' else 32
  with torch.inference_mode():
   for i in range(0,len(wanted),batch):
    px=f['pixels'][wanted[i:i+batch].tolist()]
    im=(torch.as_tensor(px,device=device).permute(0,3,1,2).float()/255-mean)/std
    z=model.projector(model.encoder(im,interpolate_pos_encoding=True).last_hidden_state[:,0]);zs.append(z.cpu())
    if i%(batch*100)==0:
     (out/'progress.json').write_text(json.dumps({'stage':'encoding','done':min(i+batch,len(wanted)),'total':len(wanted),'elapsed_seconds':time.time()-start}))
     print('ENCODING',i,len(wanted),flush=True)
  latent=torch.cat(zs);np.savez_compressed(out/'pair_indices.npz',train=train,validation=val,unique_rows=wanted)
  encoded={'checkpoint_sha256':checkpoint_sha,'pair_seed':seed,'latents':latent,
   'train_indices':torch.from_numpy(np.searchsorted(wanted,train[:,:2])),
   'validation_indices':torch.from_numpy(np.searchsorted(wanted,val[:,:2])),
   'train_labels':torch.from_numpy(train[:,2].astype(np.float32)/224),
   'validation_labels':torch.from_numpy(val[:,2].astype(np.float32)/224)}
  torch.save(encoded,cache)
  (out/'data_audit.json').write_text(json.dumps({'train_pairs':ntrain,'validation_pairs':nval,'train_episode_count':len(set(train[:,3])),'validation_episode_count':len(set(val[:,3])),'test_episode_overlap':0,'unique_encoded_rows':len(wanted),'temporal_bins':[[1,15],[16,30],[31,50],[51,75],[76,'episode end']],'train_bin_counts':tc,'val_bin_counts':vc,'sampling':'equal fixed absolute-delta bins; eligible episodes sampled uniformly within bin; random pair order','adaptation':'TRM-inspired, not exact package reproduction; strict episode train/val separation; bin edges explicitly selected here before outcomes','checkpoint':modelpath,'checkpoint_sha256':checkpoint_sha,'split_sha256':sha(R/'splits/manifest.json'),'pair_indices_sha256':sha(out/'pair_indices.npz')},indent=2))
  del model,mean,std,zs
  if device=='cuda':torch.cuda.empty_cache()
else:encoded=torch.load(cache,map_location='cpu',weights_only=False)
assert encoded['checkpoint_sha256']==checkpoint_sha
assert len(encoded['train_labels'])==ntrain and len(encoded['validation_labels'])==nval
lat=encoded['latents'].to(device);trainidx=encoded['train_indices'].to(device);validx=encoded['validation_indices'].to(device);truth=encoded['train_labels'].to(device);vtruth=encoded['validation_labels'].to(device)
results={}
for kind in ['temporal','shuffled']:
 head_path=out/(kind+'.pt')
 if (out/(kind+'_summary.json')).exists():results[kind]=json.loads((out/(kind+'_summary.json')).read_text());continue
 torch.manual_seed(seed);head=TemporalHead(lat.shape[-1]).to(device);opt=torch.optim.AdamW(head.parameters(),lr=1e-3,weight_decay=1e-4)
 y=truth.clone()
 if kind=='shuffled':
  perm=torch.randperm(len(y),generator=torch.Generator().manual_seed(seed+2),device='cpu').to(device);y=y[perm]
 generator=torch.Generator().manual_seed(seed+3);best=float('inf');history=[];fitstart=time.time()
 for epoch in range(epochs):
  head.train();order=torch.randperm(len(y),generator=generator);total=0.
  for sl in order.split(1024):
   ix=sl.to(device);pair=trainidx[ix];pred=head(lat[pair[:,0]],lat[pair[:,1]]);loss=torch.nn.functional.smooth_l1_loss(pred,y[ix])
   opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(ix)
  head.eval()
  with torch.inference_mode():
   preds=torch.cat([head(lat[p[:,0]],lat[p[:,1]]) for p in validx.split(1024)])
   val_loss=float(torch.nn.functional.smooth_l1_loss(preds,vtruth))
  entry={'epoch':epoch+1,'train_loss':total/len(y),'validation_loss':val_loss};history.append(entry)
  if val_loss<best:
   best=val_loss;best_epoch=epoch+1
   torch.save({'state_dict':head.state_dict(),'dim':lat.shape[-1],'kind':kind,'checkpoint_sha256':checkpoint_sha,'source_model':modelpath,'seed':seed,'best_epoch':best_epoch,'label_scale':224,'architecture':'4D-256-256-1 SiLU/Softplus','adapted':True},head_path)
  print('HEAD',kind,epoch+1,val_loss,flush=True)
  (out/'progress.json').write_text(json.dumps({'stage':kind,'epoch':epoch+1,'max_epochs':epochs,'best_validation_loss':best}))
 ck=torch.load(head_path,map_location=device,weights_only=False);head.load_state_dict(ck['state_dict'])
 with torch.inference_mode():preds=torch.cat([head(lat[p[:,0]],lat[p[:,1]]) for p in validx.split(1024)])
 result={'kind':kind,'epochs':epochs,'best_epoch':best_epoch,'validation_loss':best,'validation_temporal_mae_steps':float((preds-vtruth).abs().mean()*224),'validation_spearman':float(spearmanr(preds.cpu().numpy(),vtruth.cpu().numpy()).statistic),'history':history,'fit_seconds':time.time()-fitstart,'parameters':sum(p.numel() for p in head.parameters()),'selection':'minimum heldout pair loss; no navigation outcome','path':str(head_path)}
 (out/(kind+'_summary.json')).write_text(json.dumps(result,indent=2));results[kind]=result
summary={'tag':tag,'model':modelpath,'checkpoint_sha256':checkpoint_sha,'elapsed_seconds':time.time()-start,'heads':results,'complete':True,'smoke':ntrain<100000 or epochs<20,'device':device}
(out/'summary.json').write_text(json.dumps(summary,indent=2));print('HEADS_COMPLETE',tag,flush=True)
