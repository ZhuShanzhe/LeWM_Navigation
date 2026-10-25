"""Independent pre-policy geometry/visibility audit of the entire frozen val43 panel."""
import collections
import contextlib
import fcntl
import json
import math
import sys
from pathlib import Path

import numpy as np

from paper_freeze import ROOT, DATA, CHAIN, sha, frozen
from paper_common import PROFILE, CONTRACT
from paper_scope import MANIFEST, GEOMETRY, COUNT
sys.path.insert(0, str(CHAIN/'tools'))
sys.path.insert(0, str(DATA/'tools'))
from instance_eval_goat import opensim
from nav_build import pathlength, setpose


def main():
    manifest = json.loads(MANIFEST.read_text())
    assert all(sha(Path(p)) == d for p, d in manifest['source_files'].items())
    panels = {}
    for task, info in manifest['panels'].items():
        p = Path(info['path']); assert sha(p) == info['sha256']
        panels[task] = [json.loads(x) for x in p.read_text().splitlines()]
    failures = []; records = []
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from chain_gpu_idle import wait
        wait()
        for scene in manifest['scene_ids']:
            output = ROOT/'results/geometry_compact430'/f'{scene}.json'
            if output.exists():
                cached = json.loads(output.read_text())
                assert cached['freeze_sha256'] == sha(MANIFEST)
                assert cached['contract_sha256'] == sha(ROOT/'protocol/common_v1.json')
                records.extend(cached['records']); failures.extend(cached['failures']); continue
            sim = opensim(scene, PROFILE)
            local = []; issues = []
            try:
                targets = {t['instance_id']: t for t in json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text())}
                for task, episodes in panels.items():
                    for e in [e for e in episodes if e['scene_id'] == scene]:
                        try:
                            assert sim.pathfinder.is_navigable(e['start_position'], max_y_delta=.05)
                            instance = task == 'instance_image_nav'
                            views = e['valid_approach_views'] if instance else [{'position': e['goal_position']}]
                            ds = [pathlength(sim, e['start_position'], v['position']) for v in views]
                            ds = [d for d in ds if d is not None and math.isfinite(d)]
                            assert ds and abs(min(ds)-e['geodesic_m']) < .02
                            r = {'episode_id': e['episode_id'], 'scene_id': scene, 'task': task,
                                'geodesic_verified': True, 'horizontal_visibility_verified': None,
                                'discrete_head_visibility_verified': None}
                            if instance:
                                target = targets[e['instance_id']]
                                level = tilted = 0.
                                for view in views:
                                    if not sim.pathfinder.is_navigable(view['position'], max_y_delta=.05):
                                        continue
                                    obs = setpose(sim, view['position'], view['yaw'])
                                    level = max(level, float((obs['semantic'] == target['semantic_id']).mean()))
                                    center = (np.asarray(target['aabb_min'])+np.asarray(target['aabb_max']))/2
                                    p = np.asarray(view['position'])
                                    pitch = math.atan2(center[1]-(p[1]+.88), np.linalg.norm(center[[0, 2]]-p[[0, 2]]))
                                    pitch = np.clip(round(pitch/(math.pi/6)), -2, 2)*math.pi/6
                                    obs = setpose(sim, view['position'], view['yaw'], pitch)
                                    tilted = max(tilted, float((obs['semantic'] == target['semantic_id']).mean()))
                                    if level >= .025 and tilted >= .025:
                                        break
                                r.update(horizontal_visibility_verified=level >= .025,
                                    discrete_head_visibility_verified=max(level, tilted) >= .025,
                                    max_level_fraction=level, max_discrete_pitch_fraction=tilted)
                            local.append(r)
                        except Exception as exc:
                            issues.append({'episode_id': e['episode_id'], 'error': type(exc).__name__, 'detail': str(exc)})
                # Check canonical action geometry independently of policy outcomes.
                probe = next(e for e in panels['place_image_nav'] if e['scene_id'] == scene)
                for action in ['turn_left', 'turn_right', 'look_up', 'look_down']:
                    setpose(sim, probe['start_position'], probe['start_yaw'])
                    before = np.asarray(sim.get_agent(0).get_state().position).copy()
                    sim.step(action)
                    assert np.linalg.norm(np.asarray(sim.get_agent(0).get_state().position)-before) < 1e-5
            finally:
                sim.close()
            result = {'freeze_sha256': sha(MANIFEST),
                'contract_sha256': sha(ROOT/'protocol/common_v1.json'), 'records': local, 'failures': issues}
            frozen(output, result); records.extend(local); failures.extend(issues)
            print(json.dumps({'scene': scene, 'episodes_checked': len(records), 'engineering_issues': len(failures)}), flush=True)
    result = {'passed': len(records) == COUNT*2 and not failures, 'episodes_checked': len(records),
        'scenes': len({r['scene_id'] for r in records}), 'engineering_issues': failures,
        'instance_horizontal_verified': sum(r['horizontal_visibility_verified'] is True for r in records),
        'instance_head_pitch_verified': sum(r['discrete_head_visibility_verified'] is True for r in records),
        'visibility_not_verified_is_not_unreachable_proof': True,
        'panel_filtered_or_resampled': False, 'policy_calls': 0,
        'scope': 'Frozen geometric paths and finite reference-pose visibility audit, not policy performance',
        'freeze_sha256': sha(MANIFEST),
        'contract_sha256': sha(ROOT/'protocol/common_v1.json')}
    frozen(GEOMETRY, result)
    print(json.dumps(result), flush=True)
    assert result['passed']


if __name__ == '__main__':
    main()
