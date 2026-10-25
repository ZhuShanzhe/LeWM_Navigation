"""Score-isolated common simulator RPC for image-place and exact-instance tasks."""
import argparse
import contextlib
import json
import math
import sys
import time
import os
from pathlib import Path

# Keep a dedicated JSON channel before Habitat/C++ imports. Redirect BOTH Python
# and native fd1 diagnostics to stderr; contextlib alone cannot redirect C++.
PROTOCOL_OUTPUT = os.fdopen(os.dup(sys.stdout.fileno()), 'w', buffering=1)
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())

import numpy as np
from PIL import Image

from paper_freeze import ROOT, DATA, CHAIN
from paper_common import PROFILE
sys.path.insert(0, str(CHAIN/'tools'))
sys.path.insert(0, str(DATA/'tools'))
from instance_eval_goat import opensim, sensor_values, write
from nav_build import setpose, pathlength
from habitat_sim.utils.common import quat_to_coeffs

parser = argparse.ArgumentParser()
parser.add_argument('--panel', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--buffer', required=True)
parser.add_argument('--max-steps', type=int, default=500)
args = parser.parse_args()
assert 1 <= args.max_steps <= 500
panel = [json.loads(line) for line in Path(args.panel).read_text().splitlines()]
out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
buffer = Path(args.buffer)
sim = None; scene = None; index = -1; terminal = False; records = []


def transmit():
    state = sim.get_agent(0).get_state()
    gps, compass = sensor_values(state.position, state.rotation, origin, start_rotation)
    # UniGoal's graph and BEV retain their native normalization convention /30.
    # Physical sensor range is clipped to5m for every common track.
    depth = np.clip(obs['depth'], 0, 5)[..., None].astype(np.float32)/30.
    np.savez(buffer, rgb=obs['rgb'][:, :, :3], depth=depth,
        gps=np.asarray(gps, np.float32), compass=np.asarray([compass], np.float32),
        position=state.position, rotation_xyzw=quat_to_coeffs(state.rotation), goal_rgb=goal)
    info = {'time': step, 'sensor_pose': [0., 0., 0.],
        'agent_height': .88+float(state.position[1]-origin[1]), 'episode_no': index,
        'success': None, 'spl': None, 'distance_to_goal': None}
    if terminal:
        r = records[-1]
        info.update(success=r['success'], spl=r['spl'], distance_to_goal=r['final_distance_m'])
    print(json.dumps({'buffer': str(buffer), 'done': terminal, 'info': info}), file=PROTOCOL_OUTPUT, flush=True)


try:
    for line in sys.stdin:
        request = json.loads(line)
        if request['op'] == 'close':
            break
        if request['op'] == 'reset':
            if terminal and index+1 == len(panel):
                transmit(); continue
            index += 1
            assert index < len(panel)
            e = panel[index]
            if scene != e['scene_id']:
                if sim is not None:
                    sim.close()
                scene = e['scene_id']
                with contextlib.redirect_stdout(sys.stderr):
                    sim = opensim(scene, PROFILE)
                    import habitat_sim as hs
                    import magnum as mn
                    spec = hs.CameraSensorSpec()
                    spec.uuid = 'depth'; spec.sensor_type = hs.SensorType.DEPTH
                    spec.resolution = PROFILE['resolution']; spec.hfov = mn.Deg(79)
                    spec.position = mn.Vector3(0., .88, 0.)
                    sim.add_sensor(spec)
            instance = e['task'] == 'instance_image_nav'
            targets = json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text()) if instance else []
            target = next(t for t in targets if t['instance_id'] == e['instance_id']) if instance else None
            goal = np.asarray(Image.open(DATA/'datasets/hssd_image_nav_v1'/e['goal_image']).convert('RGB'))
            obs = setpose(sim, e['start_position'], e['start_yaw'])
            origin = np.asarray(sim.get_agent(0).get_state().position).copy()
            start_rotation = sim.get_agent(0).get_state().rotation
            assert sim.pathfinder.is_navigable(origin, max_y_delta=.05)
            approaches = [v['position'] for v in e['valid_approach_views']] if instance else [e['goal_position']]
            def distance(position):
                values = [pathlength(sim, position, p) for p in approaches]
                return min((d for d in values if d is not None and np.isfinite(d)), default=math.inf)
            initial_distance = distance(origin)
            assert math.isfinite(initial_distance)
            step = 0; length = 0.; terminal = False; started = time.monotonic()
            trace = (out/(e['episode_id']+'.steps.jsonl')).open('x')
            transmit()
            continue
        assert request['op'] == 'step' and not terminal
        action = int(request['action']); assert action in range(6)
        state = sim.get_agent(0).get_state()
        before = np.asarray(state.position).copy(); q0 = quat_to_coeffs(state.rotation).tolist()
        expected = before.copy()
        if action == 1:
            import quaternion
            desired = before+quaternion.rotate_vectors(state.rotation, np.array([0., 0., -.25]))
            expected = np.asarray(sim.pathfinder.try_step_no_sliding(before, desired)).copy()
        if action:
            obs = sim.step({1: 'move_forward', 2: 'turn_left', 3: 'turn_right', 4: 'look_up', 5: 'look_down'}[action])
        if action == 1:
            # Raw habitat_sim exposes the native flag in the returned observation;
            # previous_step_collided belongs to the Habitat-Lab wrapper, not this API.
            assert isinstance(obs['collided'], (bool, np.bool_))
        after = np.asarray(sim.get_agent(0).get_state().position).copy()
        final_state = sim.get_agent(0).get_state()
        camera_state = final_state.sensor_states['rgb']
        assert np.linalg.norm(after-expected) < 1e-4
        length += float(np.linalg.norm(after-before))
        fraction = float((obs['semantic'] == target['semantic_id']).mean()) if instance else None
        row = {'t': step, 'action': action, 'before': before.tolist(), 'after': after.tolist(),
            'rotation_xyzw_before': q0, 'navmesh_expected_after': expected.tolist(),
            'rotation_xyzw_after': quat_to_coeffs(final_state.rotation).tolist(),
            'camera_position_after': camera_state.position.tolist(),
            'camera_rotation_xyzw_after': quat_to_coeffs(camera_state.rotation).tolist(),
            'target_fraction_after': fraction, 'distance_after_m': distance(after),
            'collided': bool(obs['collided']) if action == 1 else False,
            'global_abort': bool(request.get('aborted', False)), 'step_wall_unix_s': time.time()}
        trace.write(json.dumps(row)+'\n'); trace.flush(); step += 1
        write(out/'progress.json', {'steps': step, 'updated_unix_s': time.time()})
        terminal = action == 0 or step >= args.max_steps
        if terminal:
            d = distance(after)
            success = int(action == 0 and not row['global_abort'] and d < (.5 if instance else 1.)
                and (not instance or fraction >= .025))
            result = {'episode_id': e['episode_id'], 'scene_id': scene, 'task': e['task'],
                'status': 'complete', 'steps': step, 'goal_stop': action == 0 and not row['global_abort'],
                'global_abort': row['global_abort'], 'success': success,
                'spl': success*initial_distance/max(initial_distance, length, 1e-8),
                'initial_distance_m': initial_distance, 'final_distance_m': d,
                'path_length_m': length, 'target_fraction_at_end': fraction,
                'forced_budget_termination': action != 0 and step >= args.max_steps,
                'episode_wall_seconds': time.monotonic()-started, 'category_supplied': False}
            records.append(result); write(out/(e['episode_id']+'.json'), result)
            write(out/'records.json', records); trace.close()
        transmit()
except Exception as exc:
    import traceback
    traceback.print_exc(file=sys.stderr)
    write(out/'runtime_failure.json', {'status': 'technical_failure', 'error': type(exc).__name__,
        'detail': str(exc), 'unfinished_counted_as_failure': False})
    print(json.dumps({'error': type(exc).__name__, 'detail': str(exc)}), file=PROTOCOL_OUTPUT, flush=True)
    raise
finally:
    if sim is not None:
        sim.close()
