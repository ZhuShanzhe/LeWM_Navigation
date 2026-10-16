"""Restricted inspection of publisher GOAT checkpoints, never arbitrary pickle."""
import builtins
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import typing
import torch

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')

class Metadata:
    """Opaque data container; serialized configuration behavior is discarded."""
    pass

def metadata_subscription(*args):
    # Only serialized typing annotations use getitem. No operand behavior runs.
    return Metadata

def safe_load(path):
    aliases = []
    for name in torch.serialization.get_unsafe_globals_in_checkpoint(path):
        if name.startswith(('habitat.config.default_structured_configs.',
                            'habitat_baselines.config.default_structured_configs.',
                            'omegaconf.', 'goat_bench.config.', 'goat_bench.obs_transformer.resize.')):
            value = Metadata
        elif name in {'builtins.dict', 'builtins.list', 'builtins.float', 'builtins.int',
                      'builtins.str', 'builtins.bool', 'builtins.Ellipsis'}:
            value = getattr(builtins, name.split('.')[-1])
        elif name in {'typing.Dict', 'typing.List', 'typing.Tuple', 'typing.Any'}:
            value = getattr(typing, name.split('.')[-1])
        elif name == 'collections.defaultdict':
            value = defaultdict
        elif name == '_operator.getitem':
            value = metadata_subscription
        else:
            raise ValueError('Unreviewed serialized global: ' + name)
        aliases.append((value, name))
    with torch.serialization.safe_globals(aliases):
        return torch.load(path, map_location='cpu', weights_only=True)

def decode(obj):
    if isinstance(obj, Metadata):
        d = obj.__dict__
        if '_content' in d:
            return decode(d['_content'])
        if '_val' in d:
            return decode(d['_val'])
        return {k: decode(v) for k, v in d.items() if k not in {'_parent', '_metadata'}}
    if isinstance(obj, dict):
        return {str(k): decode(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [decode(v) for v in obj]
    if obj is None or isinstance(obj, (str, bool, int, float)):
        return obj
    return str(type(obj).__name__)

if __name__ == '__main__':
    for path in sorted((ROOT / 'weights').glob('goat_*.pth')):
        value = safe_load(path)
        state = value['state_dict']
        result = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                  'state_keys': {k: list(v.shape) for k, v in state.items()},
                  'config': decode(value.get('config')), 'checkpoint_top_keys': list(value)}
        output = ROOT / 'results' / (path.name + '.audit.json')
        output.write_text(json.dumps(result, indent=2))
        print(json.dumps({'checkpoint': path.name, 'state_keys': len(state), 'audit': str(output)}), flush=True)
