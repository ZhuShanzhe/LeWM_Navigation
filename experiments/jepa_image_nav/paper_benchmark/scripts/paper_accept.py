"""Independent trace/metric consistency gate; full-run physical replay remains required."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from paper_freeze import ROOT, sha, frozen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode-folder', required=True)
    parser.add_argument('--configuration', required=True)
    parser.add_argument('--max-actions', type=int, default=500)
    args = parser.parse_args()
    out = Path(args.episode_folder)
    assert not (out/'runtime_failure.json').exists()
    config = json.loads(Path(args.configuration).read_text())
    assert config['contract_sha256'] == sha(ROOT/'protocol/common_v1.json')
    panel = out/'panel.jsonl'
    episodes = [json.loads(line) for line in panel.read_text().splitlines()]
    assert len(episodes) == 1
    e = episodes[0]
    records = json.loads((out/'records.json').read_text()); assert len(records) == 1
    r = records[0]; assert r['status'] == 'complete' and r['episode_id'] == e['episode_id']
    steps = [json.loads(line) for line in (out/(e['episode_id']+'.steps.jsonl')).read_text().splitlines()]
    assert len(steps) == r['steps'] and 1 <= len(steps) <= args.max_actions
    assert np.linalg.norm(np.asarray(steps[0]['before'])-np.asarray(e['start_position'])) < 1e-4
    length = 0.
    for i, s in enumerate(steps):
        assert s['t'] == i and 0 <= s['action'] < 6
        p = np.asarray(s['before'], np.float32); q = np.asarray(s['after'], np.float32)
        length += float(np.linalg.norm(q-p))
        if i:
            assert np.linalg.norm(p-np.asarray(steps[i-1]['after'])) < 1e-5
        if s['action'] != 1:
            assert np.linalg.norm(q-p) < 1e-5
        assert np.linalg.norm(q-np.asarray(s['navmesh_expected_after'])) < 1e-4
        assert s['action'] != 0 or i+1 == len(steps)
        qa = np.asarray(s['rotation_xyzw_before']); qb = np.asarray(s['rotation_xyzw_after'])
        if s['action'] in [2, 3]:
            angle = 2*math.acos(float(np.clip(abs(np.dot(qa, qb)), -1, 1)))
            assert abs(angle-math.pi/6) < 1e-4
    assert length == r['path_length_m']
    last = steps[-1]
    stopped = last['action'] == 0 and not last['global_abort']
    assert r['goal_stop'] == stopped
    instance = e['task'] == 'instance_image_nav'
    success = int(stopped and last['distance_after_m'] < (.5 if instance else 1.)
        and (not instance or last['target_fraction_after'] >= .025))
    assert success == r['success']
    assert abs(r['final_distance_m']-last['distance_after_m']) < 1e-7
    expected_spl = success*r['initial_distance_m']/max(r['initial_distance_m'], length, 1e-8)
    assert abs(expected_spl-r['spl']) < 1e-8
    policy = out/'policy_actions.jsonl'
    if policy.exists():
        actions = [json.loads(x) for x in policy.read_text().splitlines()]
        assert len(actions) == len(steps)
        for a, s in zip(actions, steps):
            assert a['canonical_action'] == s['action']
            assert a['native_action'] == int(np.argmax(a['logits']))
    elif config['model'] != 'unigoal':
        raise AssertionError('Missing native policy action evidence')
    result = {'acceptance': 'passed', 'scope': 'Per-episode trace consistency, not full physical replay',
        'episodes': 1, 'trace_steps_checked': len(steps), 'successes': success,
        'independent_physics_replay_complete': False, 'records': records,
        'contract_sha256': config['contract_sha256'], 'episode_panel_sha256': sha(panel),
        'trace_sha256': sha(out/(e['episode_id']+'.steps.jsonl'))}
    frozen(out/'accepted_summary.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
