"""Freeze43x10 per task as a prefix of the previously frozen stratified43x30 panel."""
import collections
import json
from pathlib import Path
from paper_freeze import ROOT, sha, frozen
from paper_scope import COUNT, PER_SCENE, MANIFEST, panel_path


def main():
    parent_path = ROOT/'manifests/hssd_val43_freeze.json'
    parent = json.loads(parent_path.read_text())
    assert len(parent['scene_ids']) == 43
    manifest = {k: v for k, v in parent.items() if k != 'panels'}
    manifest.update(version=2, per_scene_per_task=PER_SCENE, panels={},
        parent_manifest_sha256=sha(parent_path), parent_manifest_path=str(parent_path),
        selection_rule='First10 in each scene of original deterministic stratified selection order; no model scores',
        human_scope_authorization='2026-10-17 compact two-image-task benchmark',
        formal_model_runs=1720, official_gibson_runs=4901,
        no_additional_large_category_language_motion_runs=True)
    original_hashes = {str(parent_path): sha(parent_path)}
    for task, suffix in [('place_image_nav', 'place'), ('instance_image_nav', 'instance')]:
        entry = parent['panels'][task]
        source = Path(entry['path']); assert sha(source) == entry['sha256']
        original_hashes[str(source)] = sha(source)
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        assert len(rows) == 1290
        counts = collections.Counter(); chosen = []
        for row in rows:
            if counts[row['scene_id']] < PER_SCENE:
                chosen.append(row); counts[row['scene_id']] += 1
        assert set(counts) == set(parent['scene_ids']) and set(counts.values()) == {PER_SCENE}
        assert len(chosen) == COUNT and len({r['episode_id'] for r in chosen}) == COUNT
        dest = panel_path(suffix)
        frozen(dest, ''.join(json.dumps(r, sort_keys=True)+'\n' for r in chosen).encode())
        scenes = {}
        for scene in parent['scene_ids']:
            group = [e for e in chosen if e['scene_id'] == scene]
            scenes[scene] = {'episodes': len(group),
                'distance': dict(collections.Counter(e['distance_bin'] for e in group)),
                'detour': dict(collections.Counter(e['detour_bin'] for e in group)),
                'unique_goal_images': len({e['goal_image'] for e in group}),
                'unique_target_instances': len({e['instance_id'] for e in group}) if suffix == 'instance' else None}
        manifest['panels'][task] = {'path': str(dest), 'sha256': sha(dest), 'episodes': COUNT, 'scenes': scenes}
    frozen(MANIFEST, manifest)
    old = json.loads((ROOT/'manifests/legacy_completed_files.json').read_text())
    assert all(sha(Path(p)) == h for p, h in old['files'].items())
    assert all(sha(Path(p)) == h for p, h in original_hashes.items())
    result = {'passed': True, 'formal_model_runs': 1720, 'scenes': 43,
        'per_task': COUNT, 'per_scene_per_task': PER_SCENE, 'policy_calls': 0, 'paid_requests': 0,
        'selection_reads_model_results': False, 'parent_panel_unchanged': True,
        'legacy_completed_files_unchanged': len(old['files']), 'freeze_sha256': sha(MANIFEST)}
    frozen(ROOT/'results/compact430_cpu_gate.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
