"""Inference-only official FGPrompt computation with a JSON-line image interface.

Simulator and GPU policy have separate Python environments. Imported network
function/class bodies are unchanged. Habitat registration, unused sensors, and
training dependencies are not required. All checkpoint state keys load strictly.
Only weights-only PyTorch deserialization is used, with a narrow, audited config
data-container alias; no arbitrary pickle loading is allowed.
"""
import argparse, ast, contextlib, copy, hashlib, json, logging, numbers, sys, time, types, typing, builtins
from pathlib import Path
from typing import Dict, Iterable, Optional, Union, List, Tuple
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn.utils.rnn import PackedSequence
import torchvision
from torchvision.transforms import ColorJitter
from gym import spaces
from einops import rearrange
from yacs.config import CfgNode

ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')

class OpaqueConfig:
    """Data-only container for ignored legacy configuration metadata."""
    pass

def safe_checkpoint(path):
    unsafe=set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    aliases=[]
    for name in sorted(unsafe):
        if name=='habitat.config.default.Config':aliases.append((CfgNode,name))
        elif name.startswith(('habitat.config.default_structured_configs.','habitat_baselines.config.default_structured_configs.','omegaconf.')):
            # Do not import serialized classes: discard their behavior entirely.
            # The final policy is built from reviewed YAML, never this metadata.
            aliases.append((OpaqueConfig,name))
        elif name in {'builtins.dict','builtins.list','builtins.float','builtins.int','builtins.str','builtins.bool'}:
            aliases.append((getattr(builtins,name.split('.')[-1]),name))
        elif name in {'typing.Dict','typing.List','typing.Any'}:aliases.append((getattr(typing,name.split('.')[-1]),name))
        elif name=='collections.defaultdict':aliases.append((__import__('collections').defaultdict,name))
        else:raise ValueError('Unreviewed pickle global: '+name)
    # Legacy Config is a yacs mapping. Deserialize into the installed data-only
    # CfgNode container, not an external class or a downloaded pickle function.
    with torch.serialization.safe_globals(aliases):
        return torch.load(path,map_location='cpu',weights_only=True)

def defs(path,ns,names=None):
    tree=ast.parse(path.read_text())
    body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)]
    for node in tree.body:
        if isinstance(node,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)) and (names is None or node.name in names):body.append(node)
        elif isinstance(node,(ast.Assign,ast.AnnAssign)) and names is None:body.append(node)
    tree=ast.Module(body=body,type_ignores=[]);ast.fix_missing_locations(tree)
    exec(compile(tree,str(path),'exec'),ns)
    return ns

class Critic(nn.Module):
    def __init__(self,n):super().__init__();self.fc=nn.Linear(n,1)
    def forward(self,x):return self.fc(x)

class Categorical(nn.Module):
    def __init__(self,n,a):super().__init__();self.linear=nn.Linear(n,a)
    def forward(self,x):return torch.distributions.Categorical(logits=self.linear(x))

class NetPolicy(nn.Module):
    def __init__(self,net,action_space,*unused):
        super().__init__();self.net=net;self.action_distribution=Categorical(net.output_size,action_space.n);self.critic=Critic(net.output_size)

def fgprompt(checkpoint):
    repo=ROOT/'repos/FGPrompt'
    logger=logging.getLogger('fgprompt');logger.setLevel(logging.WARNING)
    ns=dict(globals(),logger=logger,TensorDict=dict,F=torchvision.transforms.functional,partial=__import__('functools').partial)
    defs(ROOT/'repos/running_mean_and_var.py',dict(ns,distrib=torch.distributed))
    norm=defs(ROOT/'repos/running_mean_and_var.py',dict(ns,distrib=torch.distributed))['RunningMeanAndVar']
    rnn=defs(ROOT/'repos/rnn_state_encoder.py',ns.copy())['build_rnn_state_encoder']
    transforms=defs(repo/'src/transforms.py',dict(ns,Box=spaces.Box))
    fast=types.SimpleNamespace(**defs(repo/'src/resnet.py',dict(ns,F=torch.nn.functional)))
    enc_ns=dict(ns,RunningMeanAndVar=norm,ResizeCenterCropper=transforms['ResizeCenterCropper'],fast_resnet=fast,
                d3_40_colors_rgb=np.zeros((40,3),dtype=np.uint8))
    encoder=defs(repo/'src/models.py',enc_ns)['EarlyFuseCNNEncoder']
    sensor_names={'GibsonImageGoalFeatureSensor':'gibson_imagegoal_feature','ImageGoalSensorV2':'imagegoal_sensor_v2',
                  'QueriedImageSensor':'queried_image_sensor','InstanceImageGoalSensor':'instance_imagegoal'}
    policy_ns=dict(ns,EarlyFuseCNNEncoder=encoder,build_rnn_state_encoder=rnn,Net=nn.Module,NetPolicy=NetPolicy,
                   ResizeCenterCropper=transforms['ResizeCenterCropper'],ResizeRandomCropper=transforms['ResizeRandomCropper'],
                   baseline_registry=types.SimpleNamespace(register_policy=lambda cls:cls))
    policy_ns.update({k:type(k,(),{'cls_uuid':v}) for k,v in sensor_names.items()})
    cls=defs(repo/'src/policy.py',policy_ns,{'Mlp','Flatten','NavNetPolicy','NavNet'})['NavNetPolicy']
    space=spaces.Dict({k:spaces.Box(0,255,(256,256,3),np.uint8) for k in ['rgb','imagegoal_sensor_v2']})
    policy=cls(space,spaces.Discrete(4),hidden_size=128,num_recurrent_layers=2,rnn_type='GRU',backbone='fast_resnet9',
               goal_backbone='none',normalize_visual_inputs=True,obs_transform=transforms['ResizeCenterCropper']((128,128)),
               visual_encoder_embedding_size=512,visual_obs_inputs=['rgb','imagegoal_sensor_v2'],rgb_color_jitter=0.,
               tie_inputs_and_goal_param=False,goal_embedding_size=128,enable_feature_matching=False,cam_visual=False)
    c=safe_checkpoint(checkpoint);sd=c.get('state_dict',c)
    prefix='actor_critic.'
    sd={k[len(prefix):] if k.startswith(prefix) else k:v for k,v in sd.items()}
    policy.load_state_dict(sd,strict=True)
    policy.eval().cuda()
    return policy,c

def debit(checkpoint):
    from collections import OrderedDict
    from omegaconf import OmegaConf, DictConfig
    import gym
    sys.path.insert(0,str(ROOT/'repos/croco'))
    if not hasattr(np,'float'):np.float=float
    from models.croco import CroCoNet
    from models.croco_downstream import CroCoDownstreamBinocular
    ns=dict(globals(),F=torch.nn.functional,OrderedDict=OrderedDict,Sequential=nn.Sequential,Conv2d=nn.Conv2d,
            cast=typing.cast,Type=typing.Type,TensorDict=dict)
    norm=defs(ROOT/'repos/running_mean_and_var.py',dict(ns,distrib=torch.distributed))['RunningMeanAndVar']
    rnn=defs(ROOT/'repos/rnn_state_encoder.py',ns.copy())['build_rnn_state_encoder']
    resnet=defs(ROOT/'repos/hab_resnet.py',ns.copy())
    ns.update(RunningMeanAndVar=norm,ImageGoalSensor=types.SimpleNamespace(cls_uuid='imagegoal'))
    enc=defs(ROOT/'repos/hab_resnet_policy.py',ns.copy(),{'ResNetEncoder'})['ResNetEncoder']
    ns.update(ResNetEncoder=enc,resnet18=resnet['resnet18'],CroCoNet=CroCoNet,CroCoDownstreamBinocular=CroCoDownstreamBinocular,
              OmegaConf=OmegaConf,DictConfig=DictConfig,tf=torchvision.transforms,gym=gym,Net=nn.Module,NetPolicy=NetPolicy,
              baseline_registry=types.SimpleNamespace(register_policy=lambda cls:cls),build_rnn_state_encoder=rnn)
    cls=defs(ROOT/'repos/debit/src/debit/policy.py',ns)['DEBiTPolicy']
    cfg=OmegaConf.merge(OmegaConf.load(ROOT/'repos/debit/configs/debit/debit_defaults.yaml'),
                        OmegaConf.load(ROOT/'repos/debit/configs/debit/debit_small.yaml'))
    if 'defaults' in cfg:del cfg['defaults']
    cfg.pretrained_binoc_weights=None
    space=spaces.Dict({k:spaces.Box(0,255,(224,224,3),np.uint8) for k in ['rgb','imagegoal']})
    policy=cls(space,spaces.Discrete(4),cfg)
    c=safe_checkpoint(checkpoint);sd={k.removeprefix('actor_critic.'):v for k,v in c['state_dict'].items()}
    policy.load_state_dict(sd,strict=True);policy.eval().cuda()
    return policy,c

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',default='fgprompt');p.add_argument('--smoke',action='store_true');args=p.parse_args()
    if args.model not in {'fgprompt','debit'}:raise ValueError('Unsupported model: '+args.model)
    torch.set_num_threads(1);torch.manual_seed(20261015);np.random.seed(20261015)
    path=ROOT/'weights'/('fgprompt-early-fusion-r9.verified.pth' if args.model=='fgprompt' else 'debit_small.verified.pth')
    # Official sources contain informational prints. Keep stdout a strict protocol.
    with contextlib.redirect_stdout(sys.stderr):policy,c=(fgprompt(path) if args.model=='fgprompt' else debit(path))
    goal_key='imagegoal_sensor_v2' if args.model=='fgprompt' else 'imagegoal'
    layers=policy.net.num_recurrent_layers;size=policy.net.output_size
    def image_tensor(array):
        tensor=torch.from_numpy(array).unsqueeze(0).cuda()
        if args.model=='debit':tensor=torchvision.transforms.functional.resize(tensor.permute(0,3,1,2),[224,224],antialias=True).permute(0,2,3,1)
        return tensor
    hidden=None;goal=None;prev=0;first=True
    meta={'model':'FGPrompt-early-r9' if args.model=='fgprompt' else 'DEBiT-S','strict_state_dict':True,'checkpoint_sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),
          'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'network_source':'XinyuSun/FGPrompt' if args.model=='fgprompt' else 'naver/debit',
          'adapter':'unchanged original class/function bodies; registry and sensor metadata shims; separate simulator',
          'parameters':sum(p.numel() for p in policy.parameters())}
    (ROOT/'results'/(args.model+'_inference.json')).write_text(json.dumps(meta,indent=2))
    if args.smoke:
        obs={k:image_tensor(np.zeros((256,256,3),dtype=np.uint8)) for k in ['rgb',goal_key]}
        h=torch.zeros(1,layers,size,device='cuda')
        with torch.inference_mode():
            feat,h=policy.net(obs,h,torch.zeros((1,1),dtype=torch.long,device='cuda'),torch.zeros((1,1),dtype=torch.bool,device='cuda'))[:2]
            logits=policy.action_distribution.linear(feat)
        print(json.dumps(dict(meta,smoke_logits=logits.tolist(),hidden_shape=list(h.shape))),flush=True);return
    print(json.dumps({'ready':True,**meta}),flush=True)
    for line in sys.stdin:
        try:
            req=json.loads(line)
            if req['op']=='close':break
            if req['op']=='reset':
                goal=image_tensor(np.load(req['goal']))
                hidden=torch.zeros(1,layers,size,device='cuda');prev=0;first=True
                print(json.dumps({'ok':'reset'}),flush=True);continue
            rgb=image_tensor(np.load(req['rgb']))
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.inference_mode():
                feat,hidden=policy.net({'rgb':rgb,goal_key:goal},hidden,
                                      torch.tensor([[prev]],device='cuda'),torch.tensor([[not first]],dtype=torch.bool,device='cuda'))[:2]
                logits=policy.action_distribution.linear(feat);action=int(logits.argmax(-1).item())
            torch.cuda.synchronize();latency=(time.perf_counter()-start)*1000
            prev=action;first=False
            print(json.dumps({'action':action,'latency_ms':latency,'logits':logits[0].tolist()}),flush=True)
        except Exception as e:
            import traceback;traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error':repr(e)}),flush=True)

if __name__=='__main__':main()
