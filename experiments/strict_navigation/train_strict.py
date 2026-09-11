"""Official LeWM objective; group-disjoint episode split and train-only scaling."""
import os,sys,json,hashlib,torch
from pathlib import Path
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'strict_nav_20260910';OLD=ROOT/'round12h_20260909'
sys.path.insert(0,str(ROOT/'le-wm'))
import utils
import stable_pretraining as spt
manifest_path=Path(os.environ.get('LEWM_SPLIT_MANIFEST',str(R/'splits/manifest.json')))
manifest=json.loads(manifest_path.read_text());stats=json.loads((manifest_path.parent/'normalization.json').read_text())
folder=R/'runs'/os.environ['LEWM_RUN_TAG'];folder.mkdir(parents=True,exist_ok=True)
def normalizer(dataset,source,target):
 s=stats[source];mean=torch.tensor(s['mean'],dtype=torch.float32)[None];std=torch.tensor(s['std'],dtype=torch.float32)[None]
 return spt.data.transforms.WrapTorchTransform(utils.ZScoreNormalizer(mean,std),source=source,target=target)
utils.get_column_normalizer=normalizer
def strict_split(dataset,lengths,generator=None):
 sets={k:set(v) for k,v in manifest['episodes'].items()}
 assert all(not sets[a]&sets[b] for a,b in [('train','validation'),('train','test'),('validation','test')])
 ids={k:[i for i,(ep,start) in enumerate(dataset.clip_indices) if ep in sets[k]] for k in ['train','validation']}
 testclips=sum(ep in sets['test'] for ep,t in dataset.clip_indices)
 assert len(ids['train'])+len(ids['validation'])+testclips==len(dataset)
 audit={'manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'clip_counts':{k:len(v) for k,v in ids.items()},'excluded_test_clips':testclips,'episode_counts':manifest['episode_counts'],'normalization':'frozen train-only mean and ddof1 std','passed':True}
 (folder/'split_audit.json').write_text(json.dumps(audit,indent=2))
 return torch.utils.data.Subset(dataset,ids['train']),torch.utils.data.Subset(dataset,ids['validation'])
spt.data.random_split=strict_split
# Preserve old wrapper and official repository unchanged.
source=(OLD/'train_bounded.py').read_text().replace("work=root/'round12h_20260909'","work=root/'strict_nav_20260910'")
assert "work=root/'strict_nav_20260910'" in source
exec(compile(source,str(R/'train_strict.py'),'exec'),{'__name__':'__main__','__file__':str(R/'train_strict.py')})
