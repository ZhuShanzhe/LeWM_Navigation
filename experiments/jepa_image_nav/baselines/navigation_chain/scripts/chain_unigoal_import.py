"""Policy-side imports without Habitat's unused Magnum serialization helper.

The actual Habitat simulator runs in its original Python 3.9 environment. The
only Magnum import in UniGoal policy utilities is used by list2qua/qua2list,
which have no call sites in the publisher repository. Other utility bodies are
executed unchanged. Calling the unused helper still fails, rather than faking
a simulator or a quaternion implementation.
"""
import ast
import hashlib
import importlib
import json
from pathlib import Path
import sys
import types

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
REPO = ROOT / 'repos/UniGoal'

def prepare():
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO/'third_party/Grounded-Segment-Anything'))
    import torch
    importlib.import_module('src.utils.fmm')
    path = REPO / 'src/utils/fmm/pose_utils.py'
    tree = ast.parse(path.read_text(), filename=str(path))
    removed = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == 'magnum']
    assert len(removed) == 1 and [a.name for a in removed[0].names] == ['Quaternion']
    tree.body.remove(removed[0])
    name = 'src.utils.fmm.pose_utils'
    module = types.ModuleType(name)
    module.__file__ = str(path)
    module.__package__ = 'src.utils.fmm'
    exec(compile(tree, str(path), 'exec'), module.__dict__)
    sys.modules[name] = module
    import groundingdino._C
    from src.graph.graph import Graph
    from src.agent.unigoal.agent import UniGoal_Agent
    from src.map.bev_mapping import BEV_Map
    result = {'torch':torch.__version__, 'cuda_available':torch.cuda.is_available(),
        'groundingdino_native_cuda_extension':True,
        'policy_utilities_original_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'omitted_import':'Unused Magnum Quaternion import; list2qua is not substituted',
        'full_modules':['Graph','UniGoal_Agent','BEV_Map']}
    (ROOT / 'results/unigoal_module_import.json').write_text(json.dumps(result,indent=2))
    return Graph, UniGoal_Agent, BEV_Map

if __name__ == '__main__':
    prepare()
    print('UNIGOAL_FULL_IMPORTS_READY', flush=True)
