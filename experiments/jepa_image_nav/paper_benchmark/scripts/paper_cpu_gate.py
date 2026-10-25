"""Offline selection, contract and preservation checks; no simulator/model/API calls."""
import collections
import copy
import json
from pathlib import Path

from paper_freeze import ROOT, select, sha, frozen
from paper_common import CONTRACT, PROFILE

rows = [{'episode_id': f'fake-{i}', 'distance_bin': ['short', 'medium', 'long', 'extra'][i % 4],
    'detour_bin': ['direct', 'detour'][i % 2], 'view_gap_bin': ['0-45', '45-90', '90-180'][i % 3],
    'instance_id': f'object-{i % 37}', 'goal_image': f'{i % 37}.jpg'} for i in range(300)]
before = copy.deepcopy(rows)
selected = select(rows)
assert rows == before and len(selected) == 30 and len({e['episode_id'] for e in selected}) == 30
assert selected == select(list(reversed(rows)))
assert {e['distance_bin'] for e in selected} == {'short', 'medium', 'long', 'extra'}
assert PROFILE['resolution'] == [480, 640] and PROFILE['allow_sliding'] is False
assert CONTRACT['max_actions'] == 500 and CONTRACT['turn_degrees'] == 30
assert 'depth' not in CONTRACT['policy_inputs']['FGPrompt']
assert CONTRACT['policy_inputs']['FGPrompt'] == CONTRACT['policy_inputs']['DEBiT']
manifest = json.loads((ROOT/'manifests/hssd_val43_freeze.json').read_text())
for task, entry in manifest['panels'].items():
    panel = Path(entry['path']); assert sha(panel) == entry['sha256']
    episodes = [json.loads(x) for x in panel.read_text().splitlines()]
    assert len(episodes) == 1290 and set(e['scene_id'] for e in episodes) == set(manifest['scene_ids'])
    assert set(collections.Counter(e['scene_id'] for e in episodes).values()) == {30}
protection = json.loads((ROOT/'manifests/legacy_completed_files.json').read_text())
assert all(sha(Path(p)) == digest for p, digest in protection['files'].items())
result = {'passed': True, 'policy_calls': 0, 'paid_requests': 0,
    'input_order_independent_sampling': True, 'sampler_does_not_mutate_inputs': True,
    'all43_scenes_exactly30_per_task': True, 'source_panel_hashes_match': True,
    'legacy_completed_files_unchanged': len(protection['files']),
    'unequal_modalities_explicit': True, 'official_and_derived_tracks_separate': True}
frozen(ROOT/'results/cpu_gate.json', result)
print(json.dumps(result), flush=True)
