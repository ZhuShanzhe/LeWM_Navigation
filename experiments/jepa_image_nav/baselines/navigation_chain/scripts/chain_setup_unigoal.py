"""Full UniGoal publisher modules and compatible data-only model assets."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch
import numpy as np
from instance_cache_audit import read
from instance_fetch_assets import ROOT, get

def link(source, target):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists() or target.is_symlink():
        assert target.resolve()==source.resolve(),str(target)
        return
    target.symlink_to(source,target_is_directory=source.is_dir())

if __name__ == '__main__':
    repo = ROOT / 'repos/UniGoal'
    compat = ROOT / 'repos/Grounded-Segment-Anything-compatible'
    compat.mkdir(parents=True,exist_ok=True)
    url = 'https://raw.githubusercontent.com/IDEA-Research/Grounded-Segment-Anything/fb1f0efeab0212f59d1e909c19c9bb928585bfe6/grounded_sam_demo.py'
    raw = get(url)
    text = raw.decode()
    assert 'def load_model(model_config_path, model_checkpoint_path, device)' in text
    # The native model is SAM vit_h, not SAM-HQ. Drop an unused optional import.
    text = text.replace('    sam_hq_model_registry,\n','')
    (compat / 'grounded_sam_demo.py').write_text(text)
    record = {'source':url,'original_sha256':hashlib.sha256(raw).hexdigest(),
        'active_sha256':hashlib.sha256(text.encode()).hexdigest(),
        'compatibility_change':'Remove unused optional SAM-HQ registry import; native SAM vit_h path is unchanged',
        'dependency_version_reason':'Public helper with the exact three-argument load_model signature used by UniGoal'}
    (ROOT / 'results/unigoal_grounded_sam_compat.json').write_text(json.dumps(record,indent=2))
    link(ROOT/'repos/GroundingDINO',compat/'GroundingDINO')
    link(compat,repo/'third_party/Grounded-Segment-Anything')
    for filename in ['sam_vit_h_4b8939.pth','groundingdino_swint_ogc.pth']:
        link(ROOT/'weights'/filename,repo/'data/models'/filename)
    original=ROOT/'weights/mask_rcnn_R_50_FPN_3x.pkl'
    converted=ROOT/'weights/mask_rcnn_R_50_FPN_3x.safe.pth'
    if not converted.exists():
        value=read(original)
        assert isinstance(value,dict) and isinstance(value['model'],dict)
        tensor_value={k:torch.from_numpy(v.copy()) if isinstance(v,np.ndarray) else v for k,v in value['model'].items()}
        assert all(torch.is_tensor(v) for v in tensor_value.values())
        # Preserve publisher checkpoint alignment flags, without executable pickle.
        result={k:v for k,v in value.items() if k!='model'}
        assert all(v is None or isinstance(v,(str,int,float,bool)) for v in result.values())
        result['model']=tensor_value
        torch.save(result,converted)
        assert not torch.serialization.get_unsafe_globals_in_checkpoint(converted)
        (ROOT/'results/maskrcnn_safe_conversion.json').write_text(json.dumps({
            'original_sha256':hashlib.file_digest(original.open('rb'),'sha256').hexdigest(),
            'converted_sha256':hashlib.file_digest(converted.open('rb'),'sha256').hexdigest(),
            'tensor_count':len(tensor_value),'numpy_data_only_conversion':True,
            'alignment_flags':{k:v for k,v in result.items() if k!='model'}},indent=2))
    print(json.dumps({'compatible_publisher_modules':True,'maskrcnn_safe_weights':str(converted)}),flush=True)
