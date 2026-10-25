"""Audit author-pinned Habitat defaults as data, without executing downloaded code."""
import ast
import json
from paper_freeze import ROOT, frozen, sha
from paper_fetch_official import get

revision = '1f7cfbdd3debc825f1f2fd4b9e1a8d6d4bc9bfc7'
url = ('https://raw.githubusercontent.com/facebookresearch/habitat-lab/'+revision+
       '/habitat-lab/habitat/config/default_structured_configs.py')
path = ROOT/'official/fgprompt_native/default_structured_configs.py'
frozen(path, path.read_bytes() if path.exists() else get(url, limit=2*1024*1024))


def literal(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'field':
        values = {k.arg: k.value for k in node.keywords}
        if 'default_factory' in values and isinstance(values['default_factory'], ast.Lambda):
            return ast.literal_eval(values['default_factory'].body)
        if 'default' in values:
            return ast.literal_eval(values['default'])
    return ast.literal_eval(node)


defaults = {}
bases = {}
for node in ast.parse(path.read_text()).body:
    if isinstance(node, ast.ClassDef):
        bases[node.name] = [b.id for b in node.bases if isinstance(b, ast.Name)]
        values = {}
        for assignment in node.body:
            if isinstance(assignment, ast.AnnAssign) and isinstance(assignment.target, ast.Name) and assignment.value is not None:
                try:
                    values[assignment.target.id] = literal(assignment.value)
                except (ValueError, TypeError, SyntaxError):
                    pass
        defaults[node.name] = values
def resolved(name):
    value = {}
    for parent in bases.get(name, []):
        value.update(resolved(parent))
    value.update(defaults.get(name, {}))
    return value


agent = resolved('AgentConfig')
sliding = defaults['HabitatSimV0Config']['allow_sliding']
camera = resolved('HabitatSimRGBSensorConfig')
camera_matches = camera.get('position') == [0, 1.25, 0] and camera.get('hfov') == 90
passed = agent['height'] == 1.5 and agent['radius'] == .1 and sliding is True and camera_matches
report = {'passed': passed, 'source_url': url, 'revision': revision, 'source_sha256': sha(path),
    'agent_height': agent['height'], 'agent_radius': agent['radius'], 'allow_sliding': sliding,
    'camera_position_and_hfov_defaults_verified': camera_matches,
    'camera_resolved_defaults': camera,
    'source_executed': False, 'turn30_resolution128_budget500': 'Explicit FGPrompt policy/sensors/eval YAML overrides, not base defaults'}
frozen(ROOT/'results/fgprompt_author_defaults_gate.json', report)
print(json.dumps(report), flush=True)
assert passed
