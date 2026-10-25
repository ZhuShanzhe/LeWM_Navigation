"""Freeze a deterministic HSSD official-val panel without consulting model outputs.

This is a derived image-goal panel, not the official ObjectNav episode release.
Existing benchmark outputs and original dataset files are read-only inputs.
"""
import collections
import gzip
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path('/root/autodl-tmp/jepa_vln_paper_bench_20261017')
DATA = Path('/root/autodl-tmp/jepa_vln_data_20261014')
CHAIN = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
SEED = 'paper-val43-v1-20261017'
N = 30


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()


def frozen(path, value):
    data = value if isinstance(value, bytes) else encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data, 'Refusing to change frozen artifact: '+str(path)
    else:
        with path.open('xb') as stream:
            stream.write(data)


def rank(value):
    return hashlib.sha256((SEED+'|'+value).encode()).hexdigest()


def select(rows):
    """Balance available geometry strata and goal identities, without scoring agents."""
    buckets = collections.defaultdict(list)
    for row in rows:
        key = (row['distance_bin'], row['detour_bin'], row.get('view_gap_bin', 'not_recorded'))
        buckets[key].append(row)
    for group in buckets.values():
        group.sort(key=lambda e: rank(e['episode_id']))
    keys = sorted(buckets, key=lambda k: rank('|'.join(k)))
    chosen = []
    identities = collections.Counter()
    while len(chosen) < N:
        progress = False
        for key in keys:
            group = buckets[key]
            if not group:
                continue
            # Prefer different instances/goal images; do not silently require N objects.
            best = min(range(len(group)), key=lambda i: (
                identities[group[i].get('instance_id', group[i]['goal_image'])],
                rank(group[i]['episode_id'])))
            row = group.pop(best)
            chosen.append(row)
            identities[row.get('instance_id', row['goal_image'])] += 1
            progress = True
            if len(chosen) == N:
                break
        assert progress, 'Insufficient pre-existing valid episodes'
    return chosen


def main():
    splits_path = DATA/'raw/hssd-hab/scene_splits.yaml'
    source_splits = yaml.safe_load(splits_path.read_text())
    scenes = sorted(source_splits['val'])
    assert len(scenes) == 43 and not set(scenes) & set(source_splits['train'])
    dataset = DATA/'datasets/hssd_image_nav_v1'
    split = json.loads((dataset/'split_manifest.json').read_text())
    assert all(split['scenes'][s] == 'val' for s in scenes)
    accepted = json.loads((DATA/'reports/acceptance_20261015/full_acceptance.json').read_text())
    assert accepted['passed']
    manifest = {'version': 1, 'selection_seed': SEED, 'per_scene_per_task': N,
        'scene_ids': scenes, 'official_scene_split_sha256': sha(splits_path),
        'source_revision': split['source_revision'],
        'scope': 'Derived HSSD image-goal episodes on all43 official validation scenes; not official hidden test',
        'selection_inputs': 'Only frozen source labels, geometry strata and episode identifiers; no model outputs',
        'evaluation_only': True, 'source_files': {}, 'panels': {}}
    for scene in scenes:
        p = dataset/'instances'/f'{scene}.json'
        manifest['source_files'][str(p)] = sha(p)
    for task, suffix in [('place_image_nav', 'place'), ('instance_image_nav', 'instance')]:
        panel = []
        counts = {}
        for scene in scenes:
            source = dataset/'episodes/val'/f'{scene}.{suffix}.jsonl.gz'
            manifest['source_files'][str(source)] = sha(source)
            with gzip.open(source, 'rt') as stream:
                rows = [json.loads(line) for line in stream]
            assert all(e['split'] == 'val' and e['scene_id'] == scene and e['task'] == task for e in rows)
            chosen = select(rows)
            for e in chosen:
                e = dict(e, dataset='hssd_image_nav_v1')
                panel.append(e)
                p = dataset/e['goal_image']
                manifest['source_files'][str(p)] = sha(p)
            counts[scene] = {'episodes': len(chosen),
                'unique_goal_images': len({e['goal_image'] for e in chosen}),
                'unique_target_instances': len({e['instance_id'] for e in chosen}) if suffix == 'instance' else None,
                'distance': dict(collections.Counter(e['distance_bin'] for e in chosen)),
                'detour': dict(collections.Counter(e['detour_bin'] for e in chosen))}
        assert len(panel) == 1290 and len({e['episode_id'] for e in panel}) == 1290
        content = ''.join(json.dumps(e, sort_keys=True)+'\n' for e in panel).encode()
        path = ROOT/'panels'/f'hssd_val43_{suffix}_1290.jsonl'
        frozen(path, content)
        manifest['panels'][task] = {'path': str(path), 'sha256': sha(path), 'episodes': len(panel), 'scenes': counts}
    frozen(ROOT/'manifests/hssd_val43_freeze.json', manifest)
    # No raw trajectories are copied or published. This protects already completed scores.
    protection = ROOT/'manifests/legacy_completed_files.json'
    if not protection.exists():
        files = {}
        for name in ['goat_shared_hssd300', 'goat_native_camera_common_geometry300',
                     'goat_category_hssd300_v3', 'eb_navigation_qwen25vl7b',
                     'jeplo_native_motion_validated', 'unigoal_native_instance_hssd300',
                     'jev_native_category_hssd300']:
            folder = CHAIN/'results'/name
            candidates = list(folder.glob('accepted_summary.json'))
            candidates += list(folder.glob('episodes/*/accepted_summary.json'))
            candidates += list(folder.glob('episodes/*/record.json'))
            for p in candidates:
                files[str(p)] = sha(p)
        frozen(protection, {'scope': 'Already completed immutable results only; growing aggregate records excluded', 'files': files})
    old = json.loads(protection.read_text())
    assert all(sha(Path(p)) == digest for p, digest in old['files'].items())
    print(json.dumps({'passed': True, 'scenes': 43, 'place': 1290, 'instance': 1290,
        'freeze_sha256': sha(ROOT/'manifests/hssd_val43_freeze.json'),
        'legacy_completed_artifacts_preserved': len(old['files'])}), flush=True)


if __name__ == '__main__':
    main()
