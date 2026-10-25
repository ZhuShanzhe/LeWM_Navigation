"""Original GOAT monolithic pretrained inference with a narrow sensor RPC.

Original policy/encoder bodies are evaluated unchanged from pinned publisher
source. Habitat sensor registrations are metadata shims only. All 349 checkpoint
keys must match strictly. CLIP is constructed from reviewed original source and
safe tensor assets, without loading or executing TorchScript/pickle code.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
import time
import types
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn
from gym import spaces
from torchvision import transforms as T
from safetensors.torch import load_file

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
OLD = Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
sys.path.insert(0, str(OLD / 'tools'))
import baseline_policy as bp
from instance_checkpoint_audit import safe_load

class PolicyHeads(bp.NetPolicy):
    def __init__(self, net, action_space, policy_config=None, aux_loss_config=None):
        assert not aux_loss_config
        super().__init__(net, action_space)

def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

def create_policy():
    publisher = module('publisher_clip_model', ROOT / 'repos/CLIP/clip/model.py')
    clip_weights = ROOT / 'weights/clip_RN50_openai.safetensors'
    state = load_file(str(clip_weights))
    # build_model uses strict load_state_dict and original OpenAI fp16 conversion.
    goal_encoder = publisher.build_model(state).eval().cuda()
    goal_preprocess = T.Compose([T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(224), T.Lambda(lambda im: im.convert('RGB')), T.ToTensor(),
        T.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711))])
    rnn_ns = dict(vars(bp), TensorDict=dict)
    rnn = bp.defs(OLD / 'repos/rnn_state_encoder.py', rnn_ns)['build_rnn_state_encoder']
    registry = types.SimpleNamespace(register_policy=lambda **kwargs: lambda cls: cls)
    ns = dict(globals(), Net=nn.Module, NetPolicy=PolicyHeads,
              baseline_registry=registry, build_rnn_state_encoder=rnn,
              PointNavResNetNet=types.SimpleNamespace(PRETRAINED_VISUAL_FEATURES_KEY='visual_features'),
              clip=types.SimpleNamespace(load=lambda name: (copy.deepcopy(goal_encoder), goal_preprocess)),
              logger=logging.getLogger('goat'))
    sensors = {
        'EpisodicCompassSensor': 'compass', 'EpisodicGPSSensor': 'gps',
        'ObjectGoalSensor': 'objectgoal', 'CacheImageGoalSensor': 'cache_instance_imagegoal',
        'ClipGoalSelectorSensor': 'clip_goal_selector', 'ClipImageGoalSensor': 'clip_imagegoal',
        'ClipObjectGoalSensor': 'clip_objectgoal', 'GoatGoalSensor': 'goat_subtask_goal',
        'GoatMultiGoalSensor': 'goat_multi_goal', 'LanguageGoalSensor': 'language_goal'}
    ns.update({name: type(name, (), {'cls_uuid': uuid}) for name, uuid in sensors.items()})
    definitions = bp.defs(ROOT / 'repos/goat-bench/goat_bench/models/clip_policy.py', ns)
    obs = spaces.Dict({'rgb': spaces.Box(0, 255, (224, 224, 3), np.uint8),
        'gps': spaces.Box(-np.inf, np.inf, (2,), np.float32),
        'compass': spaces.Box(-np.inf, np.inf, (1,), np.float32),
        'goat_subtask_goal': spaces.Box(-np.inf, np.inf, (1024,), np.float32)})
    policy = definitions['GOATPolicy'](obs, spaces.Discrete(7), hidden_size=512,
                                     num_recurrent_layers=2, rnn_type='LSTM', backbone='resnet50_clip_avgpool')
    checkpoint = ROOT / 'weights/goat_checkpoints__sense_act_nn_monolithic__ckpt_best.pth'
    value = safe_load(checkpoint)
    state = value['state_dict']
    assert all(k.startswith('actor_critic.') for k in state)
    policy.load_state_dict({k[len('actor_critic.'):]: v for k, v in state.items()}, strict=True)
    policy = policy.eval().cuda()
    # Frozen visual encoder must stay fp16, exactly as original clip.load on CUDA.
    policy.net.visual_encoder.backbone.half()
    resize_ns = dict(torch=torch, Tensor=torch.Tensor, Tuple=Tuple)
    resize = bp.defs(ROOT / 'repos/goat-bench/goat_bench/obs_transformer/resize.py',
                     resize_ns, {'image_resize'})['image_resize']
    meta = {'model': 'GOAT-SenseAct-NN-monolithic', 'strict_all_state_keys': True,
        'checkpoint_keys': len(state), 'source_commit': '74c41d19d4a4c3608d1575b512087b5a529aee0e',
        'checkpoint_sha256': hashlib.file_digest(checkpoint.open('rb'), 'sha256').hexdigest(),
        'clip_sha256': hashlib.file_digest(clip_weights.open('rb'), 'sha256').hexdigest(),
        'observation_keys': sorted(obs.spaces), 'actions': ['stop', 'forward', 'left', 'right', 'look_up', 'look_down', 'subtask_stop'],
        'goal_encoder_normalization': 'raw unnormalized features, publisher caches audited',
        'deployment': 'unchanged publisher policy and encoder bodies; registration shims; independent Habitat-Sim evaluator'}
    return policy, goal_encoder, goal_preprocess, resize, meta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261016)
    with contextlib.redirect_stdout(sys.stderr):
        policy, encoder, preprocess, resize, meta = create_policy()
    if args.smoke:
        with torch.inference_mode():
            goal = encoder.encode_image(preprocess(Image.new('RGB', (256, 256))).unsqueeze(0).cuda()).float()
            observations = {'rgb': torch.zeros((1, 224, 224, 3), dtype=torch.uint8, device='cuda'),
                'goat_subtask_goal': goal, 'gps': torch.zeros((1, 2), device='cuda'),
                'compass': torch.zeros((1, 1), device='cuda')}
            hidden = torch.zeros((1, policy.net.num_recurrent_layers, 512), device='cuda')
            out, hidden = policy.net(observations, hidden, torch.zeros((1, 1), dtype=torch.long, device='cuda'),
                                     torch.zeros((1, 1), dtype=torch.bool, device='cuda'))[:2]
            logits = policy.action_distribution.linear(out)
        result = dict(meta, smoke_logits=logits[0].tolist(), hidden_shape=list(hidden.shape), smoke_pass=bool(torch.isfinite(logits).all()))
        (ROOT / 'results/goat_inference_smoke.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        return
    print(json.dumps(dict(meta, ready=True)), flush=True)
    goal = None
    category_features = None
    for line in sys.stdin:
        try:
            req = json.loads(line)
            if req['op'] == 'close':
                break
            if req['op'] == 'reset':
                if 'category' in req:
                    assert set(req) == {'op', 'category'}
                    assert req['category'] in {'chair','couch','plant','bed','toilet','tv'}
                    if category_features is None:
                        from instance_cache_audit import read
                        category_features = read(ROOT/'weights/category_name_clip_embeddings.pkl')
                    feature = np.asarray(category_features[req['category']], dtype=np.float32).reshape(1,1024)
                    assert np.isfinite(feature).all()
                    goal = torch.from_numpy(feature).cuda()
                else:
                    assert set(req) == {'op', 'goal'}
                    with torch.inference_mode():
                        goal = encoder.encode_image(preprocess(Image.fromarray(np.load(req['goal']))).unsqueeze(0).cuda()).float()
                hidden = torch.zeros((1, policy.net.num_recurrent_layers, 512), device='cuda')
                prev, first = 0, True
                print(json.dumps({'ok': 'reset'}), flush=True)
                continue
            assert set(req) <= {'op', 'rgb', 'gps', 'compass'}
            rgb = torch.from_numpy(np.load(req['rgb'])).unsqueeze(0).cuda()
            rgb = resize(rgb, (224, 224), channels_last=True, interpolation_mode='area')
            observations = {'rgb': rgb, 'goat_subtask_goal': goal,
                'gps': torch.tensor([req['gps']], dtype=torch.float32, device='cuda'),
                'compass': torch.tensor([[req['compass']]], dtype=torch.float32, device='cuda')}
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.inference_mode():
                out, hidden = policy.net(observations, hidden,
                    torch.tensor([[prev]], dtype=torch.long, device='cuda'),
                    torch.tensor([[not first]], dtype=torch.bool, device='cuda'))[:2]
                logits = policy.action_distribution.linear(out)
                action = int(logits.argmax(-1).item())
            torch.cuda.synchronize()
            prev, first = action, False
            print(json.dumps({'action': action, 'logits': logits[0].tolist(),
                              'latency_ms': (time.perf_counter()-start)*1000}), flush=True)
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            print(json.dumps({'error': type(exc).__name__}), flush=True)

if __name__ == '__main__':
    main()
