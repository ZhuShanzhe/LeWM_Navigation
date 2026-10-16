"""Numerically compare the evaluator's pose sensors to original Habitat bodies."""
import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import quaternion
from habitat_sim.utils.common import quat_from_angle_axis, quat_to_coeffs
from instance_eval_goat import ROOT, sensor_values, write

path = ROOT / 'repos/habitat_v023_nav.py'
tree = ast.parse(path.read_text())
body = [ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)]
names = {'HeadingSensor', 'EpisodicCompassSensor', 'EpisodicGPSSensor'}
for node in tree.body:
    if isinstance(node, ast.ClassDef) and node.name in names:
        node.decorator_list = []
        body.append(node)
ns = dict(np=np, quaternion=quaternion, Sensor=object,
    quaternion_rotate_vector=quaternion.rotate_vectors,
    quaternion_from_coeff=lambda xyzw: quaternion.quaternion(xyzw[3], *xyzw[:3]),
    cartesian_to_polar=lambda x, y: (math.hypot(x, y), math.atan2(y, x)))
module = ast.Module(body=body, type_ignores=[])
ast.fix_missing_locations(module)
exec(compile(module, str(path), 'exec'), ns)
rng = np.random.default_rng(20261016)
max_gps, max_compass = 0., 0.
for _ in range(1000):
    origin = rng.uniform(-10, 10, size=3)
    position = rng.uniform(-10, 10, size=3)
    start = quat_from_angle_axis(rng.uniform(-math.pi, math.pi), np.array([0., 1., 0.]))
    current = quat_from_angle_axis(rng.uniform(-math.pi, math.pi), np.array([0., 1., 0.]))
    state = SimpleNamespace(position=position, rotation=current)
    sim = SimpleNamespace(get_agent_state=lambda: state)
    episode = SimpleNamespace(start_position=origin, start_rotation=quat_to_coeffs(start))
    gps_sensor = ns['EpisodicGPSSensor'].__new__(ns['EpisodicGPSSensor'])
    gps_sensor._sim, gps_sensor._dimensionality = sim, 2
    compass_sensor = ns['EpisodicCompassSensor'].__new__(ns['EpisodicCompassSensor'])
    compass_sensor._sim = sim
    gps, compass = sensor_values(position, current, origin, start)
    native_gps = gps_sensor.get_observation({}, episode)
    native_compass = compass_sensor.get_observation({}, episode)[0]
    max_gps = max(max_gps, float(np.max(np.abs(np.asarray(gps)-native_gps))))
    max_compass = max(max_compass, abs(float(compass)-float(native_compass)))
result = {'passed': max_gps < 3e-6 and max_compass < 3e-7, 'random_cases': 1000,
    'native_source': 'facebookresearch/habitat-lab v0.2.3 habitat/tasks/nav/nav.py',
    'native_source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'maximum_gps_error_m': max_gps, 'maximum_compass_error_rad': max_compass,
    'comparison': 'Original publisher sensor method bodies versus evaluation RPC sensors'}
write(ROOT / 'results/goat_sensor_gate.json', result)
print(json.dumps(result), flush=True)
raise SystemExit(0 if result['passed'] else 1)
