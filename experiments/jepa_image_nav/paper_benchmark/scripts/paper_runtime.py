"""Freeze executable evaluation sources after interface checks and before formal scores."""
import json
from pathlib import Path
from paper_freeze import ROOT, CHAIN, sha, frozen
from paper_scope import MANIFEST

ARTIFACT = ROOT/'manifests/compact430_runtime_sources.json'


def sources():
    paths = [ROOT/'tools'/n for n in ['paper_scope.py', 'paper_common.py', 'paper_sim.py',
        'paper_eval.py', 'paper_unigoal.py', 'paper_accept.py', 'paper_replay.py', 'paper_runtime.py',
        'paper_unigoal_client_pool.py', 'paper_unigoal_mmap.py', 'paper_unigoal_call_diagnostics.py',
        'paper_unigoal_model_gate.py']]
    paths += list((CHAIN/'tools').glob('chain_unigoal*.py'))
    paths += [CHAIN/'tools/instance_policy_goat.py',
        Path('/root/autodl-tmp/jepa_vln_baselines_20261015/tools/baseline_policy.py')]
    return {str(p): sha(p) for p in sorted(set(paths))}


def verify():
    data = json.loads(ARTIFACT.read_text())
    assert data['freeze_sha256'] == sha(MANIFEST)
    assert data['contract_sha256'] == sha(ROOT/'protocol/common_v1.json')
    assert data['sources'] == sources(), 'Frozen evaluation source changed; do not silently mix results'
    return data


def freeze_runtime():
    frozen(ARTIFACT, {'sources': sources(), 'freeze_sha256': sha(MANIFEST),
        'contract_sha256': sha(ROOT/'protocol/common_v1.json'),
        'scope': 'Executable adapter and local policy-service sources after interface gates, before formal runs',
        'weights': 'Strict state-dict/checkpoint hashes independently captured by each native policy configuration'})
    verify()
