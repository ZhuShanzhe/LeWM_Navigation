"""Reuse the frozen retrieval construction with a seed-specific encoded training cache."""
import sys,json,hashlib,inspect
from pathlib import Path
import torch,numpy as np
import retrieval_prior
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
seed=int(sys.argv[1]);assert seed in [3073,3074]
source=R/'heads'/f'temporal{seed}';out=R/'priors'/f'retrieval{seed}'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((source/'summary.json').read_text())
assert summary['complete'] and not summary['smoke']
assert sha(summary['model'])==summary['checkpoint_sha256']
assert json.loads((R/'runs'/f'strict_tw_s{seed}'/'training_summary.json').read_text())['full_epoch_budget_completed']
assert not (out/'library.pt').exists(),'Use a new retry tag after auditing existing target; do not silently reuse.'
src=inspect.getsource(retrieval_prior.build)
before="source=R/'heads/temporal3072';out=R/'priors/retrieval3072'"
after=f"source=R/'heads/temporal{seed}';out=R/'priors/retrieval{seed}'"
assert src.count(before)==1
ns=dict(vars(retrieval_prior));exec(compile(src.replace(before,after),__file__,'exec'),ns);ns['build']()
library=torch.load(out/'library.pt',map_location='cpu',weights_only=False)
base=torch.load(R/'priors/retrieval3072/library.pt',map_location='cpu',weights_only=False)
assert library['checkpoint_sha256']==summary['checkpoint_sha256']
assert torch.equal(library['rows'],base['rows']) and torch.equal(library['actions'],base['actions'])
manifest=json.loads((R/'splits/manifest.json').read_text())
ep=set(library['rows'][:,2].tolist())
assert ep<=set(manifest['episodes']['train'])
assert not ep&set(manifest['episodes']['validation']) and not ep&set(manifest['episodes']['test'])
audit={'passed':True,'seed':seed,'same_training_state_pairs_and_action_chunks_as_3072':True,
 'n':len(library['rows']),'model_sha256':sha(summary['model']),'library_sha256':sha(out/'library.pt'),
 'source_builder_sha256':sha(R/'retrieval_prior.py'),'split_sha256':sha(R/'splits/manifest.json'),
 'extra_training':False,'note':'Frozen encoder differs by independently trained model; same logged training pairs/actions. No navigation outcome selection.'}
(out/'cross_seed_audit.json').write_text(json.dumps(audit,indent=2))
print('RETRIEVAL_REPEAT_BUILT',json.dumps(audit),flush=True)
