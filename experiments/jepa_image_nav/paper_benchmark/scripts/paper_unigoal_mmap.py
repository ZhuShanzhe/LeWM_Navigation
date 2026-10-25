"""Lazy private CPU storage mapping for two verified checkpoints, not model reuse."""
import contextlib
import json
import mmap
import os
import time
from pathlib import Path
from paper_freeze import ROOT, CHAIN, sha


def stamp(path):
    s = path.stat()
    return {'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'device': s.st_dev, 'inode': s.st_ino}


@contextlib.contextmanager
def mapped_checkpoints(torch):
    assert torch.serialization.get_default_mmap_options() == mmap.MAP_PRIVATE
    gate = json.loads((ROOT/'results/unigoal_mmap_load_gate_v2.json').read_text())
    assert gate['passed']
    allowed = {}
    for value in gate['files']:
        if value.get('mmap_supported') and value['file'] in {'groundingdino_swint_ogc.pth', 'sam_vit_h_4b8939.pth'}:
            allowed[(CHAIN/'weights'/value['file']).resolve()] = value
    assert len(allowed) == 2
    original = torch.load
    report = {'mapped_loads': 0, 'original_loads': 0, 'loader_seconds': 0.,
        'model_objects_reused': False, 'precision_or_weights_changed': False}
    def load(f, *args, **kwargs):
        started = time.monotonic()
        try:
            candidate = f if isinstance(f, (str, os.PathLike)) else getattr(f, 'name', None)
            path = Path(candidate).resolve() if isinstance(candidate, (str, os.PathLike)) else None
            location = args[0] if args else kwargs.get('map_location')
            supported = path in allowed and location in (None, 'cpu') and len(args) <= 1 and 'mmap' not in kwargs
            if supported and not isinstance(f, (str, os.PathLike)):
                supported = hasattr(f, 'tell') and f.tell() == 0 and not f.closed
            if supported:
                expected = allowed[path]
                assert stamp(path) == expected['file_identity'], 'Checkpoint changed after byte-equivalence validation'
                report['mapped_loads'] += 1
                return original(str(path), *args, **dict(kwargs, mmap=True))
            report['original_loads'] += 1
            return original(f, *args, **kwargs)
        finally:
            report['loader_seconds'] += time.monotonic()-started
    torch.load = load
    try:
        yield report
    finally:
        torch.load = original
