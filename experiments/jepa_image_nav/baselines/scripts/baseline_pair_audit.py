"""Paired actuator test and observed goal-region stopping upper bound."""
import collections, importlib.metadata, json, platform
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
def rows(run):
    out={}
    for file in (BASE/'results'/run).glob('*/episodes.jsonl'):
        for line in file.read_text().splitlines():
            r=json.loads(line);out[r['episode_id']]=r
    return out
if __name__=='__main__':
    a=rows('debit_gibson_val_place');b=rows('debit_native10_gibson_val_place');assert a.keys()==b.keys()
    groups=collections.defaultdict(list)
    for k in a:groups[a[k]['scene_id']].append(b[k]['success']-a[k]['success'])
    vals=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(20261015)
    ci=np.percentile(vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(1),[2.5,97.5]).tolist()
    report={'paired_episodes':len(a),'scenes':len(groups),'common_decision_budget':500,
            'debit_turn30_SR':np.mean([r['success'] for r in a.values()]),'debit_turn10_SR':np.mean([r['success'] for r in b.values()]),
            'SR_delta_turn10_minus_turn30':float(vals.mean()),'delta_scene_bootstrap_95':ci,
            'different_physical_rotation_per_decision':True,'interpretation':'Actuator adaptation effect for this checkpoint, not an algorithm improvement or a paper reproduction.'}
    (BASE/'results/actuator_pair_audit.json').write_text(json.dumps(report,indent=2))
    packages=['torch','torchvision','numpy','gym','timm','einops','omegaconf','yacs','kornia','opencv-python-headless','scikit-image','scipy']
    versions={p:importlib.metadata.version(p) for p in packages}
    (BASE/'results/policy_environment.json').write_text(json.dumps({'python':platform.python_version(),'packages':versions},indent=2))
    print(json.dumps(report),flush=True);print(versions,flush=True)
