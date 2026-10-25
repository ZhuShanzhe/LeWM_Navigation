"""Independent full-run geometry and terminal visual-score replay on frozen scenes."""
import argparse
import fcntl
import json
import math
import sys
from pathlib import Path

import numpy as np
from paper_freeze import ROOT, DATA, CHAIN, sha, frozen
from paper_common import PROFILE
from paper_scope import COUNT, panel_rows
sys.path.insert(0, str(CHAIN/'tools')); sys.path.insert(0, str(DATA/'tools'))
from instance_eval_goat import opensim
from nav_build import pathlength
from habitat_sim.utils.common import quat_from_coeffs


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', required=True)
    args = parser.parse_args(); out = ROOT/'results'/args.run
    records = []; steps_checked = 0
    contract_sha = sha(ROOT/'protocol/common_v1.json')
    folders = sorted((out/'episodes').iterdir())
    assert len(folders) == COUNT
    episodes = {}
    for f in folders:
        data = json.loads((f/'accepted_summary.json').read_text())
        assert data['acceptance'] == 'passed' and data['contract_sha256'] == contract_sha
        e = json.loads((f/'panel.jsonl').read_text())
        assert f.name == e['episode_id']
        episodes[e['episode_id']] = e
    assert len({e['scene_id'] for e in episodes.values()}) == 43
    suffix = 'instance' if next(iter(episodes.values()))['task'] == 'instance_image_nav' else 'place'
    frozen_panel = {e['episode_id']: e for e in panel_rows(suffix)}
    assert episodes == frozen_panel, 'Entire original frozen panel must be accounted without substitution'
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from chain_gpu_idle import wait
        wait()
        for scene in sorted({e['scene_id'] for e in episodes.values()}):
            dest = out/'physics_replay'/f'{scene}.json'
            if dest.exists():
                cached = json.loads(dest.read_text()); assert cached['contract_sha256'] == contract_sha
                for p, digest in cached['input_hashes'].items():
                    assert sha(Path(p)) == digest
                records.extend(cached['records']); steps_checked += cached['steps_checked']; continue
            sim = opensim(scene, PROFILE)
            targets = {t['instance_id']: t for t in json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text())}
            local = []; count = 0; hashes = {}
            try:
                import quaternion
                for f in folders:
                    if episodes[f.name]['scene_id'] != scene:
                        continue
                    e = episodes[f.name]; r = json.loads((f/'records.json').read_text())[0]
                    trace = f/(e['episode_id']+'.steps.jsonl')
                    steps = [json.loads(x) for x in trace.read_text().splitlines()]
                    hashes[str(trace)] = sha(trace); hashes[str(f/'accepted_summary.json')] = sha(f/'accepted_summary.json')
                    for step in steps:
                        p = np.asarray(step['before'], np.float32)
                        expected = p.copy()
                        if step['action'] == 1:
                            q = quat_from_coeffs(step['rotation_xyzw_before'])
                            desired = p+quaternion.rotate_vectors(q, np.asarray([0., 0., -.25]))
                            expected = np.asarray(sim.pathfinder.try_step_no_sliding(p, desired))
                        assert np.linalg.norm(expected-np.asarray(step['after'])) < 1e-4
                        count += 1
                    last = steps[-1]; instance = e['task'] == 'instance_image_nav'
                    views = [v['position'] for v in e['valid_approach_views']] if instance else [e['goal_position']]
                    initial = [pathlength(sim, e['start_position'], p) for p in views]
                    final = [pathlength(sim, last['after'], p) for p in views]
                    d0 = min(d for d in initial if d is not None and math.isfinite(d))
                    df = min(d for d in final if d is not None and math.isfinite(d))
                    assert abs(d0-r['initial_distance_m']) < 1e-4 and abs(df-r['final_distance_m']) < 1e-4
                    fraction = None
                    if instance:
                        state = sim.get_agent(0).get_state()
                        state.position = np.asarray(last['after'], np.float32)
                        state.rotation = quat_from_coeffs(last['rotation_xyzw_after'])
                        sensors = state.sensor_states
                        for pose in sensors.values():
                            pose.position = np.asarray(last['camera_position_after'], np.float32)
                            pose.rotation = quat_from_coeffs(last['camera_rotation_xyzw_after'])
                        state.sensor_states = sensors
                        sim.get_agent(0).set_state(state, infer_sensor_states=False)
                        semantic = sim.get_sensor_observations()['semantic']
                        fraction = float((semantic == targets[e['instance_id']]['semantic_id']).mean())
                        assert abs(fraction-r['target_fraction_at_end']) <= 1e-5
                    success = int(r['goal_stop'] and df < (.5 if instance else 1.) and (not instance or fraction >= .025))
                    assert success == r['success']
                    local.append({'episode_id': e['episode_id'], 'scene_id': scene,
                        'success_replayed': success, 'geometry_and_primary_score_agree': True})
            finally:
                sim.close()
            frozen(dest, {'passed': True, 'contract_sha256': contract_sha,
                'input_hashes': hashes, 'records': local, 'steps_checked': count})
            records.extend(local); steps_checked += count
            print('PHYSICS_REPLAY', args.run, scene, len(records), flush=True)
    assert len(records) == COUNT
    result = {'acceptance': 'passed', 'episodes': len(records), 'scenes': 43,
        'independent_physics_replay_complete': True, 'trace_steps_checked': steps_checked,
        'successes': sum(r['success_replayed'] for r in records), 'contract_sha256': contract_sha}
    frozen(out/'physics_replay_accepted.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
