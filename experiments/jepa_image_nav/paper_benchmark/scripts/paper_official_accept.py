"""Reconcile all original Gibson validation cases, policy decisions and native metrics."""
import gzip
import json
from pathlib import Path
import numpy as np
from paper_freeze import ROOT, sha, frozen

manifest = json.loads((ROOT/'manifests/official_gibson_imagegoal.json').read_text())
out = ROOT/'results/official_gibson_fgprompt_4901'
configuration = json.loads((out/'configuration.json').read_text())
motion_path = ROOT/'results/repair_20261019/official_motion_gate.private.json'
motion = json.loads(motion_path.read_text())
assert motion['passed'] and motion['original_pose_tolerance_unchanged']
assert motion['configuration_sha256'] == sha(out/'configuration.json')
assert motion['records_sha256'] == sha(out/'records.json')
inventory_path = ROOT/'results/repair_20261019/official_trace_inventory.private.json'
assert motion['inventory_sha256'] == sha(inventory_path)
inventory = json.loads(inventory_path.read_text())
certificates = {(e['scene'], e['episode'], e['step']): e for e in motion['certified_events']}
assert len(certificates) == len(motion['certified_events']) == len(inventory['nonforward_motion'])
observed_events = set()
expected = set()
for file in manifest['files']:
    p = Path(file['path']); assert sha(p) == file['sha256']
    data = json.loads(gzip.decompress(p.read_bytes()))
    for e in data if isinstance(data, list) else data['episodes']:
        key = (file['scene'], str(e['episode_id'])); assert key not in expected
        expected.add(key)
records = json.loads((out/'records.json').read_text())
assert {(r['scene_id'], r['episode_id']) for r in records} == expected and len(records) == len(expected) == 4901
count = 0; groups = {}; hashes = {}
for r in records:
    folder = out/'episodes'/r['scene_id']/r['episode_id']
    assert not (folder/'runtime_failure.json').exists()
    trace = folder/'steps.jsonl'; steps = [json.loads(x) for x in trace.read_text().splitlines()]
    assert sha(trace) == inventory['preserved_file_hashes'][str(trace)]
    assert sha(folder/'record.json') == inventory['preserved_file_hashes'][str(folder/'record.json')]
    assert len(steps) == r['steps'] and 1 <= len(steps) <= 500
    length = 0.
    for i, s in enumerate(steps):
        assert s['t'] == i and s['action'] == int(np.argmax(s['logits']))
        a = np.asarray(s['before'], np.float32); b = np.asarray(s['after'], np.float32)
        length += float(np.linalg.norm(b-a))
        if i:
            assert np.linalg.norm(a-np.asarray(steps[i-1]['after'])) < 1e-5
        if s['action'] != 1:
            if np.linalg.norm(b-a) >= 1e-5:
                # Native body turns also run the navmesh motion filter. A turn
                # may vertically project onto a stair surface. Accept ONLY an
                # individually certified exact native replay, not a larger tolerance.
                key = (r['scene_id'], r['episode_id'], s['t'])
                event = certificates[key]
                assert s['action'] == event['action'] and s['action'] in (2, 3)
                assert s['before'] == event['before'] and s['after'] == event['after']
                assert event['replay_after_error_m'] < 1e-5
                assert event['zero_displacement_native_filter_error_m'] < 1e-5
                observed_events.add(key)
        assert s['action'] != 0 or i+1 == len(steps)
    assert steps[-1]['action'] == r['final_action']
    final = r['native_metrics']; initial = r['initial_native_metrics']
    success = int(r['final_action'] == 0 and final['distance_to_goal'] < 1.)
    assert success == final['success']
    d0 = initial['distance_to_goal']
    spl = success*d0/max(d0, length, 1e-8)
    assert abs(spl-final['spl']) < 1e-6
    assert abs(steps[-1]['metrics']['spl']-final['spl']) < 1e-12
    groups.setdefault(r['scene_id'], []).append([success, final['spl']])
    hashes[str(trace)] = sha(trace); count += len(steps)
assert observed_events == set(certificates)
scenes = sorted(groups); means = np.asarray([np.mean(groups[s], axis=0) for s in scenes])
rng = np.random.default_rng(20261017); boot = means[rng.integers(0, len(scenes), size=(10000, len(scenes)))].mean(axis=1)
result = {'acceptance': 'passed', 'episodes': 4901, 'scenes': 14,
    'successes': int(sum(r['native_metrics']['success'] for r in records)),
    'episode_weighted_SR': float(np.mean([r['native_metrics']['success'] for r in records])),
    'episode_weighted_SPL': float(np.mean([r['native_metrics']['spl'] for r in records])),
    'scene_weighted_SR': float(means[:, 0].mean()), 'scene_weighted_SPL': float(means[:, 1].mean()),
    'scene_cluster_bootstrap95_SR': np.quantile(boot[:, 0], [.025, .975]).tolist(),
    'scene_cluster_bootstrap95_SPL': np.quantile(boot[:, 1], [.025, .975]).tolist(),
    'trace_steps_checked': count, 'original_episode_identity_coverage_verified': True,
    'configuration_sha256': sha(out/'configuration.json'), 'policy_sha256': sha(out/'policy.json'),
    'native_turn_projection_events_individually_replayed': len(observed_events),
    'native_motion_replay_certificate_sha256': sha(motion_path),
    'original_trace_bytes_and_native_metrics_preserved': True,
    'scope': 'Official original Gibson ImageGoal validation data and native task metrics; explicitly adapted Habitat0.3.1 runtime, not hidden leaderboard or bitwise author runtime'}
frozen(out/'trace_hashes.private.json', hashes)
frozen(out/'accepted_summary.json', result)
print(json.dumps(result, indent=2), flush=True)
