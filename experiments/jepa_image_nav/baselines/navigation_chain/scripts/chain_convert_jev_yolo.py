"""Extract tensors from the native YOLO checkpoint without executing pickle classes."""
import hashlib,json,sys
from pathlib import Path
import torch,yaml
from safetensors.torch import save_file
torch.set_num_threads(2)
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
source=ROOT/'weights/yolov7-e6e.pt'
assert hashlib.file_digest(source.open('rb'),'sha256').hexdigest()=='b370120a414bf32b5d65fc808e5a32c8d9b3c63902d1bc41894fc9d86eccf9cb'
class DataOnlyModule:
    def __setstate__(self,state):
        assert isinstance(state,dict)
        self.__dict__.update(state)
unsafe=torch.serialization.get_unsafe_globals_in_checkpoint(source)
allowed=[]
for name in unsafe:
    assert name.startswith(('models.common.','models.yolo.','torch.nn.modules.')),name
    cls=type('DataOnly_'+name.replace('.','_'),(DataOnlyModule,),{})
    allowed.append((cls,name))
with torch.serialization.safe_globals(allowed):
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
native=checkpoint.get('ema') or checkpoint['model']
assert isinstance(native,DataOnlyModule)
state={}
def walk(module,prefix=''):
    assert isinstance(module,DataOnlyModule)
    for key,value in module.__dict__.get('_parameters',{}).items():
        if value is not None:
            assert isinstance(value,torch.Tensor)
            state[prefix+key]=value.detach().contiguous()
    omit=module.__dict__.get('_non_persistent_buffers_set',set())
    for key,value in module.__dict__.get('_buffers',{}).items():
        if value is not None and key not in omit:
            assert isinstance(value,torch.Tensor)
            state[prefix+key]=value.detach().contiguous()
    for key,value in module.__dict__.get('_modules',{}).items():
        if value is not None:walk(value,prefix+key+'.')
walk(native)
config=native.__dict__['yaml']
matches=[]
for path in (ROOT/'repos/yolov7/cfg').rglob('*.yaml'):
    document=yaml.safe_load(path.read_text())
    if isinstance(document,dict) and all(document.get(key)==config.get(key) for key in ['nc','depth_multiple','width_multiple','anchors','backbone','head']):
        matches.append(path)
assert matches,'Checkpoint architecture does not match any reviewed publisher config'
architecture=next((p for p in matches if 'deploy' in p.parts),matches[0])
# Only the pinned publisher YAML is used to construct the native model.
sys.path.insert(0,str(ROOT/'repos/yolov7'))
from models.yolo import Model
model=Model(str(architecture),ch=3)
model.load_state_dict(state,strict=True)
path=ROOT/'weights/yolov7-e6e.safetensors'
save_file(state,str(path))
metadata={'publisher_checkpoint_sha256':hashlib.file_digest(source.open('rb'),'sha256').hexdigest(),
    'safe_weights_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
    'source':'ema' if checkpoint.get('ema') is not None else 'model',
    'architecture':str(architecture.relative_to(ROOT/'repos/yolov7')),
    'strict_state_dict_load':True,'state_keys':len(state),'pickle_class_code_executed':False,
    'names':native.__dict__['names'],'stride':native.__dict__['stride'].tolist()}
(ROOT/'results/jev_yolo_safe_conversion.json').write_text(json.dumps(metadata,indent=2))
print(json.dumps(metadata,indent=2),flush=True)
