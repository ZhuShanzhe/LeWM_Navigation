"""Verify published acceptance evidence and recompute native trajectory summaries."""
import collections,json,math
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'results'
load=lambda name:json.loads((root/name).read_text())
source=load('official_source_model_gate.json')
assert source['source_model_gate_pass'] and all(x['passed'] for x in source['checks'])
assert load('official_profile_negative_tests.json')['tests']==7
sensor=load('official_sensor_live_test.json');assert sensor['pass'] and len(sensor['cases'])==2
for case in sensor['cases']:assert case['pass'] and 8<=case['measured_publish_hz']<=12
trt=load('official_trt_fixture_test.json');assert trt['pass'] and trt['onnxruntime']=='1.22.0'
for name,model in trt['models'].items():
    assert 'TensorrtExecutionProvider' in model['providers_executed']
    threshold=.1 if name=='policy' else .5
    assert model['errors'] and all(e['finite'] and e['max_abs']<=threshold and e['mean_abs']<=.1 for e in model['errors'])
runtime=load('official_runtime_inventory.json')
assert runtime['cuda_runtime_version']==12080 and runtime['tensorrt_version']==100900 and runtime['cudnn_version']//10000==9
build=load('official_native_build_manifest.json');assert build['source_controller_unchanged']
guard=load('official_native_guard_test.json');assert guard['pass'] and all(x['rejected_before_launch'] for x in guard['tests'])
assert build['artifacts']['source/deployment/go2_deploy/deploy/src/deploy.cpp']==source['resources_sha256']['deployment/go2_deploy/deploy/src/deploy.cpp']
native=load('official_native_probes.json');assert len(native['cases'])==6 and not native['paper_performance_reproduced']
rows=collections.defaultdict(list)
for line in (root/'official_native_trajectories.jsonl').read_text().splitlines():
    row=json.loads(line);rows[row['run']].append(row)
assert set(rows)=={r['run'] for r in native['cases']}
def close(a,b):assert math.isclose(a,b,rel_tol=1e-8,abs_tol=1e-8),(a,b)
for case in native['cases']:
    seq=rows[case['run']];first,last=seq[0],seq[-1];duration=last['wall_seconds']-first['wall_seconds'];m=case['recomputed']
    assert duration>0 and all(seq[i+1]['wall_seconds']>seq[i]['wall_seconds'] for i in range(len(seq)-1))
    displacement=math.dist(first['base_xyz'][:2],last['base_xyz'][:2])
    close(m['base_xy_displacement_m'],displacement);close(m['mean_displacement_speed_m_s'],displacement/duration)
    close(m['mean_yaw_rate_rad_s'],(last['unwrapped_yaw_rad']-first['unwrapped_yaw_rad'])/duration)
    close(m['min_base_z_m'],min(r['base_xyz'][2] for r in seq));close(m['max_base_z_m'],max(r['base_xyz'][2] for r in seq))
    close(m['min_base_upright_cos'],min(r['upright_cos'] for r in seq))
    assert case['controller_cpp_fixture_pass']
print(json.dumps({'pass':True,'native_records':len(native['cases']),'trajectory_samples':sum(map(len,rows.values())),'paper_reproduction_claim':False}))
