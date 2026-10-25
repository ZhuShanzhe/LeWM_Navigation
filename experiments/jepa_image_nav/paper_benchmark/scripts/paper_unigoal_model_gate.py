"""Isolated native-vs-mmap actual-model initialization and fixed-image CUDA regression."""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from paper_freeze import ROOT, CHAIN, DATA, frozen, sha


def tensor_hash(value):
    import torch
    x = value.detach().cpu().contiguous()
    return hashlib.sha256(x.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def worker(variant, trial):
    import numpy as np
    import torch
    import yaml
    from PIL import Image
    sys.path.insert(0, str(CHAIN/'tools'))
    from chain_unigoal_import import REPO, prepare
    Graph, _, _ = prepare()
    os.chdir(REPO)
    from groundingdino.util import get_tokenlizer
    tok = get_tokenlizer.get_tokenlizer; bert = get_tokenlizer.get_pretrained_language_model
    get_tokenlizer.get_tokenlizer = lambda name: tok(str(CHAIN/'models/bert-base-uncased') if name == 'bert-base-uncased' else name)
    get_tokenlizer.get_pretrained_language_model = lambda name: bert(str(CHAIN/'models/bert-base-uncased') if name == 'bert-base-uncased' else name)
    config = yaml.safe_load((REPO/'configs/config_habitat.yaml').read_text())
    config.update(map_size=config['map_size_cm']//config['map_resolution'], device=torch.device('cuda:0'),
        env_frame_width=640, env_frame_height=480, hfov=79,
        base_url='http://127.0.0.1:11435/v1/', api_key='ollama',
        llm_model='llama3.2-vision:11b', vlm_model='llama3.2-vision:11b')
    # This seed controls only this no-navigation regression, not formal policy decoding.
    torch.manual_seed(1); np.random.seed(1); random.seed(1)
    initial_cpu_rng = tensor_hash(torch.get_rng_state()); initial_cuda_rng = tensor_hash(torch.cuda.get_rng_state())
    from paper_unigoal_mmap import mapped_checkpoints
    context = mapped_checkpoints(torch) if variant == 'mmap' else contextlib.nullcontext({})
    with context as loading:
        started = time.monotonic(); graph = Graph(SimpleNamespace(**config)); torch.cuda.synchronize()
        setup_seconds = time.monotonic()-started
        post_cpu_rng = tensor_hash(torch.get_rng_state()); post_cuda_rng = tensor_hash(torch.cuda.get_rng_state())
        modules = {'GroundingDINO': graph.grounded_sam[0], 'SAM': graph.grounded_sam[1].model,
            'DISK': graph.extractor, 'LightGlue': graph.image_matcher}
        states = {name: {key: {'dtype': str(t.dtype), 'shape': list(t.shape), 'sha256': tensor_hash(t)}
            for key, t in module.state_dict().items()} for name, module in modules.items()}
        episode = json.loads((ROOT/'panels/hssd_val43_instance_compact430.jsonl').read_text().splitlines()[0])
        image_path = DATA/'datasets/hssd_image_nav_v1'/episode['goal_image']
        image = Image.open(image_path).convert('RGB'); pixels = np.asarray(image)
        import GroundingDINO.groundingdino.datasets.transforms as transforms
        transform = transforms.Compose([transforms.RandomResize([800], max_size=1333), transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        tensor, _ = transform(image, None)
        started = time.monotonic()
        with torch.no_grad():
            output = graph.grounded_sam[0].to('cuda')(tensor.to('cuda')[None], captions=[graph.node_space+'.'])
            dino = {k: tensor_hash(output[k]) for k in ['pred_logits', 'pred_boxes']}
            graph.grounded_sam[1].set_image(pixels)
            sam = tensor_hash(graph.grounded_sam[1].get_image_embedding())
        torch.cuda.synchronize(); forward_seconds = time.monotonic()-started
    value = {'variant': variant, 'initial_CPU_RNG': initial_cpu_rng, 'initial_CUDA_RNG': initial_cuda_rng,
        'post_initialization_CPU_RNG': post_cpu_rng, 'post_initialization_CUDA_RNG': post_cuda_rng,
        'all_model_states': states, 'GroundingDINO_outputs': dino, 'SAM_image_embedding': sam,
        'input_image_sha256': sha(image_path), 'setup_seconds': setup_seconds,
        'fixed_image_forward_seconds': forward_seconds, 'loader': loading,
        'navigation_or_LLM_calls': 0, 'statistical_performance_score': False}
    frozen(ROOT/'results/unigoal_loader_equivalence'/f'{variant}_{trial}.json', value)
    print(json.dumps({k: v for k, v in value.items() if k != 'all_model_states'}), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--variant', choices=['native', 'mmap'])
    parser.add_argument('--trial', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    if args.variant:
        worker(args.variant, args.trial); return
    dest = ROOT/'results/unigoal_actual_model_load_gate.json'
    if dest.exists():
        assert json.loads(dest.read_text())['passed']; return
    sys.path.insert(0, str(CHAIN/'tools'))
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from chain_gpu_idle import wait
        wait()
        # Reverse the order in pair2 to limit warm-cache/first-run timing bias.
        for trial, variant in [(0, 'native'), (0, 'mmap'), (1, 'mmap'), (1, 'native')]:
            out = ROOT/'results/unigoal_loader_equivalence'/f'{variant}_{trial}.json'
            if out.exists():
                continue
            with (ROOT/'logs'/f'unigoal_model_load_{variant}_{trial}.log').open('a') as log:
                subprocess.run([sys.executable, '-u', __file__, '--variant', variant, '--trial', str(trial)], check=True,
                    stdout=log, stderr=subprocess.STDOUT)
        fields = ['initial_CPU_RNG', 'initial_CUDA_RNG', 'post_initialization_CPU_RNG',
            'post_initialization_CUDA_RNG', 'all_model_states', 'GroundingDINO_outputs', 'SAM_image_embedding', 'input_image_sha256']
        pairs = []
        for trial in [0, 1]:
            original = json.loads((ROOT/'results/unigoal_loader_equivalence'/f'native_{trial}.json').read_text())
            mapped = json.loads((ROOT/'results/unigoal_loader_equivalence'/f'mmap_{trial}.json').read_text())
            assert all(original[k] == mapped[k] for k in fields), 'Actual model loading or forward pass differs'
            assert mapped['loader']['mapped_loads'] == 2
            baseline = original['setup_seconds']+original['fixed_image_forward_seconds']
            optimized = mapped['setup_seconds']+mapped['fixed_image_forward_seconds']
            pairs.append({'native_seconds': baseline, 'mapped_seconds': optimized, 'saved_seconds': baseline-optimized})
        result = {'passed': True, 'bit_exact_model_states_and_fixed_image_outputs': True,
            'CPU_and_CUDA_RNG_states_equal': True, 'order_balanced_pairs': pairs,
            'activate_mmap': all(p['saved_seconds'] > 1. for p in pairs), 'activation_threshold_seconds_each_pair': 1.,
            'small_regression_sample_not_a_precise_throughput_estimate': True,
            'adapter_sha256': sha(ROOT/'tools/paper_unigoal_mmap.py'),
            'load_gate_sha256': sha(ROOT/'results/unigoal_mmap_load_gate_v2.json'),
            'native_and_mmap_report_sha256': {f'{v}_{t}': sha(ROOT/'results/unigoal_loader_equivalence'/f'{v}_{t}.json')
                for v in ['native', 'mmap'] for t in [0, 1]},
            'scope': 'Actual frozen checkpoint model initialization and fixed-image CUDA output regression, no task scores'}
        frozen(dest, result); print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
