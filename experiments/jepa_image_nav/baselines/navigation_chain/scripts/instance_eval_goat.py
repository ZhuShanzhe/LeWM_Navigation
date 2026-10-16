"""GOAT single-subtask HSSD evaluation, separate from official HM3D scores.

Ground truth is confined to this evaluator. The pretrained policy receives RGB,
goal RGB and native episodic GPS/compass only. Native camera, seven actions and
subtask STOP semantics are preserved. Frozen goal photographs are unchanged.
"""
import argparse
import concurrent.futures as cf
from collections import Counter
import fcntl
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
DATA = Path('/root/autodl-tmp/jepa_vln_data_20261014')
OLD = Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
sys.path.insert(0, str(DATA / 'tools'))

PROFILES = {
    'native': {'resolution': [640, 360], 'hfov': 42, 'camera_height': 1.31,
               'radius': .17, 'height': 1.41, 'climb': .1, 'cell_height': .05, 'allow_sliding': False},
    'shared': {'resolution': [256, 256], 'hfov': 90, 'camera_height': 1.25,
               'radius': .18, 'height': 1.4, 'allow_sliding': True},
}
PROFILES['native_camera_common_geometry'] = dict(PROFILES['shared'],
    resolution=[640,360], hfov=42, camera_height=1.31)

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)

def sensor_values(position, rotation, origin, start_rotation):
    import quaternion
    relative = quaternion.rotate_vectors(start_rotation.inverse(), np.asarray(position)-origin)
    gps = [-float(relative[2]), float(relative[0])]
    heading = quaternion.rotate_vectors(rotation.inverse()*start_rotation, np.array([0., 0., -1.]))
    compass = math.atan2(float(heading[0]), -float(heading[2]))
    return gps, compass

def opensim(scene, profile):
    import habitat_sim as hs
    import magnum as mn
    cfg = hs.SimulatorConfiguration()
    cfg.scene_dataset_config_file = str(DATA / 'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json')
    cfg.scene_id, cfg.enable_physics = scene, True
    cfg.allow_sliding = profile['allow_sliding']
    agent = hs.agent.AgentConfiguration()
    agent.height, agent.radius = profile['height'], profile['radius']
    sensors = []
    for name, kind in [('rgb', hs.SensorType.COLOR), ('semantic', hs.SensorType.SEMANTIC)]:
        spec = hs.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = profile['resolution']
        spec.position = mn.Vector3(0, profile['camera_height'], 0)
        spec.hfov = mn.Deg(profile['hfov'])
        sensors.append(spec)
    agent.sensor_specifications = sensors
    agent.action_space = {name: hs.agent.ActionSpec(name, hs.agent.ActuationSpec(amount=amount))
        for name, amount in [('move_forward', .25), ('turn_left', 30), ('turn_right', 30), ('look_up', 30), ('look_down', 30)]}
    sim = hs.Simulator(hs.Configuration(cfg, [agent]))
    sim.seed(20261016)
    nav = hs.NavMeshSettings()
    nav.set_defaults()
    nav.agent_radius, nav.agent_height = profile['radius'], profile['height']
    nav.include_static_objects = True
    if 'climb' in profile:
        nav.agent_max_climb, nav.cell_height = profile['climb'], profile['cell_height']
    if not sim.recompute_navmesh(sim.pathfinder, nav):
        sim.close()
        raise RuntimeError('Navmesh reconstruction failed')
    for index, (_, obj) in enumerate(sorted(sim.get_rigid_object_manager().get_objects_by_handle_substring('').items()), 1):
        obj.semantic_id = index
    return sim

def scene_eval(args):
    from nav_build import setpose, pathlength
    from habitat_sim.utils.common import quat_to_coeffs
    folder = ROOT / 'results' / args.run / args.scene
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        panel_bytes = Path(args.panel).read_bytes()
        episodes = [json.loads(line) for line in panel_bytes.decode().splitlines() if line.strip()]
        episodes = [e for e in episodes if e['scene_id'] == args.scene]
        if args.limit:
            episodes = episodes[:args.limit]
        config = {'panel_sha256': hashlib.sha256(panel_bytes).hexdigest(), 'scene': args.scene,
            'profile': PROFILES[args.profile], 'profile_name': args.profile, 'max_steps': args.max_steps,
            'episodes': len(episodes), 'task': 'single_instance_image_goal',
            'scope': 'HSSD derived cross-domain single-subtask evaluation, not official GOAT-Bench/HM3D reproduction',
            'goal_stop_action': 6, 'abort_action': 0, 'project_success': 'subtask_stop and geodesic_to_reference_approach<0.5m and target_pixels>=2.5%',
            'viewpoint_success': 'subtask_stop and geodesic_to_reference_approach<0.25m; GOAT configuration threshold on HSSD-derived labels',
            'policy_inputs': ['goal_rgb', 'rgb', 'episodic_gps', 'episodic_compass'],
            'score_only': ['navmesh', 'instance_id', 'semantic_id', 'target_pose', 'approach_views', 'success_distance']}
        if (folder / 'config.json').exists():
            assert json.loads((folder / 'config.json').read_text()) == config
        else:
            write(folder / 'config.json', config)
        if (folder / 'summary.json').exists():
            return json.loads((folder / 'summary.json').read_text())
        dataset = DATA / 'datasets/hssd_image_nav_v1'
        targets = {x['instance_id']: x for x in json.loads((dataset / 'instances' / (args.scene + '.json')).read_text())}
        sim = opensim(args.scene, PROFILES[args.profile])
        log = (folder / 'policy_stderr.log').open('a')
        process = subprocess.Popen([str(ROOT / 'goat_env/bin/python'), str(ROOT / 'tools/instance_policy_goat.py')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, bufsize=1)
        poll = selectors.DefaultSelector()
        poll.register(process.stdout, selectors.EVENT_READ)
        def rpc(request=None):
            if request is not None:
                process.stdin.write(json.dumps(request)+'\n')
                process.stdin.flush()
            if not poll.select(timeout=90):
                raise TimeoutError('Policy response timeout')
            line = process.stdout.readline()
            if not line:
                raise RuntimeError('Policy exited; see policy_stderr.log')
            result = json.loads(line)
            if 'error' in result:
                raise RuntimeError('Policy error: ' + result['error'])
            return result
        try:
            policy = rpc()
            assert policy['ready'] and policy['strict_all_state_keys']
            write(folder / 'policy.json', policy)
            with tempfile.TemporaryDirectory(prefix='goat_eval_', dir='/dev/shm') as temporary:
                tmp = Path(temporary)
                for e in episodes:
                    dest = folder / (e['episode_id'] + '.json')
                    if dest.exists():
                        continue
                    goal = np.asarray(Image.open(dataset / e['goal_image']).convert('RGB'))
                    np.save(tmp / 'goal.npy', goal)
                    rpc({'op': 'reset', 'goal': str(tmp / 'goal.npy')})
                    observations = setpose(sim, e['start_position'], e['start_yaw'])
                    initial = sim.get_agent(0).get_state()
                    origin, start_rotation = np.array(initial.position), initial.rotation
                    if args.profile != 'native' and not sim.pathfinder.is_navigable(origin, max_y_delta=.05):
                        raise RuntimeError('Frozen episode start is not navigable under the declared profile')
                    def distance(position):
                        values = [pathlength(sim, position, v['position']) for v in e['valid_approach_views']]
                        return min((d for d in values if d is not None and np.isfinite(d)), default=1e6)
                    initial_distance = distance(origin)
                    if initial_distance >= 1e6:
                        write(dest, {'episode_id': e['episode_id'], 'scene_id': args.scene,
                                     'status': 'invalid_under_profile_navmesh', 'success': 0})
                        continue
                    length, collisions, forwards, visible_ever, near_ever = 0., 0, 0, False, False
                    minimum = initial_distance
                    actions, latencies = [], []
                    target = targets[e['instance_id']]
                    with gzip.open(folder / (e['episode_id'] + '.steps.jsonl.gz'), 'wt') as trace:
                        for step in range(args.max_steps):
                            state = sim.get_agent(0).get_state()
                            before = np.array(state.position)
                            d = distance(before)
                            fraction = float((observations['semantic'] == target['semantic_id']).mean())
                            visible_ever |= fraction >= .025
                            near_ever |= bool(d < .5 and fraction >= .025)
                            minimum = min(minimum, d)
                            gps, compass = sensor_values(before, state.rotation, origin, start_rotation)
                            np.save(tmp / 'rgb.npy', observations['rgb'][:, :, :3])
                            decision = rpc({'op': 'act', 'rgb': str(tmp / 'rgb.npy'), 'gps': gps, 'compass': compass})
                            action = int(decision['action'])
                            assert 0 <= action <= 6
                            actions.append(action)
                            latencies.append(decision['latency_ms'])
                            trace.write(json.dumps({'t': step, 'action': action, 'position': before.tolist(),
                                'rotation_xyzw': quat_to_coeffs(state.rotation).tolist(), 'gps': gps, 'compass': compass,
                                'distance_m': d, 'target_fraction': fraction, 'logits': decision['logits']})+'\n')
                            if action in {0, 6}:
                                break
                            name = {1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'look_up', 5: 'look_down'}[action]
                            observations = sim.step(name)
                            after = np.array(sim.get_agent(0).get_state().position)
                            length += float(np.linalg.norm(after-before))
                            if action == 1:
                                forwards += 1
                                collisions += int(bool(observations['collided']))
                    final = sim.get_agent(0).get_state()
                    d = distance(final.position)
                    fraction = float((observations['semantic'] == target['semantic_id']).mean())
                    stopped = actions[-1] == 6
                    success = bool(stopped and d < .5 and fraction >= .025)
                    viewpoint_success = bool(stopped and d < .25)
                    efficiency = initial_distance / max(initial_distance, length, 1e-8)
                    near_ever |= bool(d < .5 and fraction >= .025)
                    visible_ever |= fraction >= .025
                    record = {'episode_id': e['episode_id'], 'scene_id': args.scene, 'status': 'complete',
                        'category': e['category'], 'task': e['task'], 'distance_bin': e['distance_bin'],
                        'steps': len(actions), 'success': int(success), 'spl': float(success)*efficiency,
                        'viewpoint_success_025': int(viewpoint_success), 'viewpoint_spl_025': float(viewpoint_success)*efficiency,
                        'final_distance_m': d, 'initial_distance_m': initial_distance, 'minimum_distance_m': min(minimum, d),
                        'path_length_m': length, 'target_fraction_at_end': fraction, 'target_visible_ever': visible_ever,
                        'reached_success_region': near_ever, 'goal_stop': stopped, 'global_abort': actions[-1] == 0,
                        'action_counts': dict(Counter(actions)), 'collision_rate': collisions/max(forwards, 1),
                        'inference_ms_mean': float(np.mean(latencies)), 'inference_ms_p95': float(np.percentile(latencies, 95)),
                        'outcome': ('success' if success else 'global_abort' if actions[-1] == 0 else
                            'reached_but_failed_to_stop' if near_ever else 'premature_subtask_stop' if stopped else
                            'collision_dominated_timeout' if collisions/max(forwards, 1) > .5 else 'search_or_localization_timeout')}
                    write(dest, record)
                    print(json.dumps({'episode': e['episode_id'], 'success': int(success), 'steps': len(actions)}), flush=True)
            records = [json.loads((folder / (e['episode_id'] + '.json')).read_text()) for e in episodes]
            summary = dict(config, status='complete', completed=sum(x['status']=='complete' for x in records),
                invalid_under_profile=sum(x['status']!='complete' for x in records),
                successes=sum(x.get('success', 0) for x in records),
                SR=float(np.mean([x.get('success', 0) for x in records])),
                SPL=float(np.mean([x.get('spl', 0.) for x in records])),
                outcomes=dict(Counter(x.get('outcome', x['status']) for x in records)))
            write(folder / 'summary.json', summary)
            return summary
        finally:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=15)
            poll.close()
            log.close()
            sim.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', default=str(OLD / 'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl'))
    parser.add_argument('--run', required=True)
    parser.add_argument('--scene')
    parser.add_argument('--profile', choices=PROFILES, default='native')
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.scene:
        scene_eval(args)
    else:
        episodes = [json.loads(line) for line in Path(args.panel).read_text().splitlines() if line.strip()]
        scenes = sorted({e['scene_id'] for e in episodes})
        def child(scene):
            log = ROOT / 'logs' / (args.run+'_'+scene+'.log')
            env = dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, __file__, '--panel', args.panel, '--run', args.run, '--scene', scene,
                       '--profile', args.profile, '--max-steps', str(args.max_steps), '--limit', str(args.limit)]
            with log.open('w') as output:
                result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, env=env)
            return {'scene': scene, 'exit_code': result.returncode}
        with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(child, scenes))
        write(ROOT / 'results' / args.run / 'execution.json', results)
        raise SystemExit(int(any(r['exit_code'] for r in results)))
