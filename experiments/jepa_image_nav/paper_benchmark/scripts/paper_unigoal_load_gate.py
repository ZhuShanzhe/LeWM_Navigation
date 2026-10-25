"""CPU checkpoint loader audit. Test identical values and RNG with mmap; no GPU use."""
import collections
import gc
import hashlib
import json
import time
import zipfile
from pathlib import Path

import torch
from paper_freeze import ROOT, CHAIN, sha, frozen
from paper_unigoal_mmap import stamp


def tensor_inventory(value, prefix=''):
    if isinstance(value, torch.Tensor):
        cpu = value.detach().cpu().contiguous()
        raw = cpu.reshape(-1).view(torch.uint8).numpy().tobytes()
        return [(prefix, str(value.dtype), list(value.shape), hashlib.sha256(raw).hexdigest())]
    if isinstance(value, dict):
        result = []
        for k in sorted(value, key=lambda x: (str(type(x)), str(x))):
            result.extend(tensor_inventory(value[k], prefix+'/'+str(k)))
        return result
    if isinstance(value, (list, tuple)):
        return [r for i, x in enumerate(value) for r in tensor_inventory(x, prefix+'/'+str(i))]
    # Preserve metadata types/values too, not just floating-point weights.
    return [(prefix, type(value).__name__, repr(value))]


def main():
    paths = [CHAIN/'weights'/n for n in ['groundingdino_swint_ogc.pth',
        'sam_vit_h_4b8939.pth', 'mask_rcnn_R_50_FPN_3x.safe.pth', 'disk_lightglue_v0-1_arxiv.pth']]
    results = []
    for path in paths:
        if not zipfile.is_zipfile(path):
            results.append({'file': path.name, 'mmap_supported': False,
                'action': 'Keep original loader unchanged; legacy serialization'})
            continue
        digest = sha(path); initial = torch.get_rng_state().clone()
        start = time.monotonic(); normal = torch.load(path, map_location='cpu')
        normal_load = time.monotonic()-start
        assert torch.equal(initial, torch.get_rng_state())
        before = tensor_inventory(normal); del normal; gc.collect()
        start = time.monotonic(); mapped = torch.load(path, map_location='cpu', mmap=True)
        mmap_load = time.monotonic()-start
        assert torch.equal(initial, torch.get_rng_state())
        after = tensor_inventory(mapped)
        assert before == after and sha(path) == digest
        del mapped; gc.collect()
        results.append({'file': path.name, 'sha256': digest, 'mmap_supported': True,
            'file_identity': stamp(path),
            'all_tensors_and_metadata_bit_identical': True, 'CPU_RNG_unchanged': True,
            'tensor_entries': sum(len(v) == 4 for v in before),
            'normal_load_seconds': normal_load, 'mmap_metadata_load_seconds': mmap_load,
            'timing_limit': 'mmap is lazy; this timing excludes later page faults/copies, not end-to-end speedup'})
        print(json.dumps(results[-1]), flush=True)
    report = {'passed': True, 'model_inference_calls': 0, 'paid_requests': 0,
        'GPU_used': False, 'torch_version': torch.__version__, 'files': results,
        'native_loader_defaults_preserved': True,
        'runtime_activation_requires_actual_model_load_and_forward_equivalence': True}
    frozen(ROOT/'results/unigoal_mmap_load_gate_v2.json', report)
    print('CPU_LOAD_GATE_PASSED', flush=True)


if __name__ == '__main__':
    main()
