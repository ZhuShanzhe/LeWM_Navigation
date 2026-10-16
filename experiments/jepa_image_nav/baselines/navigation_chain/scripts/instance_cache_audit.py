"""Restricted NumPy data-only reader for publisher goal-cache protocol audits."""
import json
import pickle
from collections import OrderedDict
from pathlib import Path
import numpy as np

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')

class ArrayOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if module in {'numpy.core.multiarray', 'numpy._core.multiarray'} and name in {'_reconstruct', 'scalar'}:
            return getattr(np.core.multiarray, name)
        if module == 'numpy' and name in {'ndarray', 'dtype'}:
            return getattr(np, name)
        raise pickle.UnpicklingError('Unreviewed cache global: ' + module + '.' + name)

def read(path):
    with path.open('rb') as stream:
        return ArrayOnlyUnpickler(stream, encoding='latin1').load()

def arrays(value, key=''):
    if isinstance(value, np.ndarray):
        yield key, value
    elif isinstance(value, dict):
        for k, v in value.items():
            yield from arrays(v, key + '/' + str(k))
    elif isinstance(value, (list, tuple)):
        for i, v in enumerate(value):
            yield from arrays(v, key + '/' + str(i))

if __name__ == '__main__':
    result = {}
    for filename in ['1S7LAXRdDqK_CLIP_goat_embedding.pkl', 'category_name_clip_embeddings.pkl']:
        value = read(ROOT / 'weights' / filename)
        features = [(k, a) for k, a in arrays(value) if a.size == 1024]
        norms = [float(np.linalg.norm(a.astype(float))) for _, a in features]
        result[filename] = {'top_keys': list(value)[:5], 'embeddings': len(features),
            'sample_shape': list(features[0][1].shape), 'norm_min': min(norms),
            'norm_max': max(norms), 'norm_median': float(np.median(norms)),
            'unit_normalized': bool(np.allclose(norms, 1., atol=1e-3))}
    (ROOT / 'results/goat_goal_cache_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
