"""Explicit compact evaluation scope; original frozen panels remain immutable."""
import json
from paper_freeze import ROOT, sha

COUNT = 430
PER_SCENE = 10
MANIFEST = ROOT/'manifests/hssd_val43_compact430_freeze.json'
GEOMETRY = ROOT/'results/geometry_compact430_gate.json'
TRACKS = [('fgprompt', 'place'), ('debit', 'place'), ('goat', 'instance')]


def panel_path(task):
    assert task in ('place', 'instance')
    return ROOT/'panels'/f'hssd_val43_{task}_compact430.jsonl'


def run_name(model, task, smoke=False):
    assert (model, task) in TRACKS+[('unigoal', 'instance')]
    return f'{model}_{task}_compact430'+('_smoke' if smoke else '')


def panel_rows(task):
    manifest = json.loads(MANIFEST.read_text())
    entry = manifest['panels'][task+'_image_nav']
    path = panel_path(task)
    assert entry['path'] == str(path) and entry['sha256'] == sha(path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == COUNT and len({e['episode_id'] for e in rows}) == COUNT
    return rows


def require_geometry():
    gate = json.loads(GEOMETRY.read_text())
    assert gate['passed'] and gate['freeze_sha256'] == sha(MANIFEST)
    assert gate['contract_sha256'] == sha(ROOT/'protocol/common_v1.json')
    return gate
