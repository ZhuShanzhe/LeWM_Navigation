import json,platform,subprocess,hashlib,importlib.metadata as md
from pathlib import Path
import torch
import stable_worldmodel as swm
root=Path('/root/autodl-tmp/lewm_research')
model=swm.wm.utils.load_pretrained(str(root/'data/hf_tworoom'))
versions={p:md.version(p) for p in ['torch','torchvision','numpy','stable-worldmodel','stable-pretraining','transformers','huggingface-hub','hydra-core','omegaconf','h5py','hdf5plugin','gymnasium','lightning','imageio','imageio-ffmpeg']}
result={'python':platform.python_version(),'platform':platform.platform(),'gpu':torch.cuda.get_device_name(),'gpu_memory_bytes':torch.cuda.get_device_properties(0).total_memory,
 'torch_cuda':torch.version.cuda,'versions':versions,
 'git_commit':subprocess.check_output(['git','-C',str(root/'le-wm'),'rev-parse','HEAD'],text=True).strip(),
 'git_status':subprocess.check_output(['git','-C',str(root/'le-wm'),'status','--short'],text=True),
 'official_checkpoint_parameters':sum(p.numel() for p in model.parameters()),
 'official_checkpoint_strict_loading':True,
 'files':{}}
for p in [root/'env.sh',root/'run_eval.py',root/'run_train.py',root/'data/checkpoints/train_pilot500/weights_epoch_5.pt']:
 result['files'][str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
(root/'reports/provenance.json').write_text(json.dumps(result,indent=2))
(root/'reports/requirements.freeze.txt').write_text(subprocess.check_output([str(root/'venv/bin/python'),'-m','pip','freeze'],text=True))
print(json.dumps(result,indent=2))
