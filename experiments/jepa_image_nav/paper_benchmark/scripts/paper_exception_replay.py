"""Replay only valid UniGoal episodes, preserving the original physics/score kernel.

No navigation is rerun. Requires all original frozen cases to be accounted as
valid or separately evidenced native implementation exceptions. A new private
certificate explicitly reports coverage, never claims 430 valid completions.
"""
import argparse
import ast
import fcntl
import json
import math
import sys
from pathlib import Path

import numpy as np
from paper_freeze import ROOT, DATA, CHAIN, sha, frozen
from paper_common import PROFILE
from paper_scope import panel_rows, run_name
from paper_exception_audit import audit, require
from paper_runtime import verify


def replay_kernel(tree):
    """Extract the unchanged original geometry and primary-score statements."""
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    loops = [n for f in functions for n in ast.walk(f) if isinstance(n, ast.For)
             and isinstance(n.target, ast.Name) and n.target.id == 'f'
             and isinstance(n.iter, ast.Name) and n.iter.id == 'folders'
             and any(isinstance(s,ast.For) and isinstance(s.target,ast.Name)
                     and s.target.id=='step' for s in ast.walk(n))]
    require(len(loops) == 1, 'Ambiguous episode replay kernel')
    body = loops[0].body
    start = next(i for i,n in enumerate(body) if isinstance(n, ast.For)
                 and isinstance(n.target,ast.Name) and n.target.id == 'step')
    end = next(i for i,n in enumerate(body) if isinstance(n,ast.Assert)
               and ast.dump(n.test) == ast.dump(ast.parse("success == r['success']", mode='eval').body))
    require(start < end, 'Invalid physics kernel region')
    return [ast.dump(n,include_attributes=False) for n in body[start:end+1]]


def kernel_gate():
    verify()
    original = ROOT/'tools/paper_replay.py'; candidate = Path(__file__)
    require(replay_kernel(ast.parse(original.read_text())) ==
            replay_kernel(ast.parse(candidate.read_text())), 'Replay kernel differs from frozen original')
    result = dict(passed=True, original_replay_sha256=sha(original),
        exception_aware_replay_sha256=sha(candidate),
        geometry_terminal_visibility_success_kernel_AST_identical=True,
        policy_calls=0, GPU_calls=0, tolerances_and_scoring_unchanged=True)
    frozen(ROOT/'results/exception_replay_kernel_gate.json',result)
    return result


def replay():
    kernel_gate()
    coverage = audit(require_complete=True)
    out = ROOT/'results'/run_name('unigoal','instance')
    frozen(out/'exception_accounting_accepted.private.json',coverage)
    valid = set(coverage['episode_ids']['valid'])
    require(valid, 'No valid episodes; do not issue empty performance acceptance')
    episodes = {e['episode_id']:e for e in panel_rows('instance') if e['episode_id'] in valid}
    require(set(episodes) == valid, 'Non-frozen valid subset')
    folders = [out/'episodes'/eid for eid in sorted(valid)]
    contract_sha = sha(ROOT/'protocol/common_v1.json')
    sys.path.insert(0,str(CHAIN/'tools'));sys.path.insert(0,str(DATA/'tools'))
    from instance_eval_goat import opensim
    from nav_build import pathlength
    from habitat_sim.utils.common import quat_from_coeffs
    records=[];steps_checked=0;scene_cert_hashes={}
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        from chain_gpu_idle import wait
        wait()
        for scene in sorted({e['scene_id'] for e in episodes.values()}):
            dest = out/'physics_replay_valid'/f'{scene}.json'
            if dest.exists():
                cached=json.loads(dest.read_text())
                require(cached['contract_sha256']==contract_sha,'Replay contract changed')
                require(cached['coverage_sha256']==sha(out/'exception_accounting_accepted.private.json'),
                        'Replay cached against different coverage')
                require(all(sha(Path(p))==h for p,h in cached['input_hashes'].items()),
                        'Replay inputs changed')
                expected={e['episode_id'] for e in episodes.values() if e['scene_id']==scene}
                require({r['episode_id'] for r in cached['records']}==expected,'Cached subset changed')
                records.extend(cached['records']);steps_checked+=cached['steps_checked']
                scene_cert_hashes[str(dest)]=sha(dest);continue
            sim = opensim(scene, PROFILE)
            targets = {t['instance_id']: t for t in json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text())}
            local=[];count=0;hashes={}
            try:
                import quaternion
                for f in folders:
                    if episodes[f.name]['scene_id'] != scene:
                        continue
                    e=episodes[f.name];r=json.loads((f/'records.json').read_text())[0]
                    trace=f/(e['episode_id']+'.steps.jsonl')
                    steps=[json.loads(x) for x in trace.read_text().splitlines()]
                    for p in (trace,f/'accepted_summary.json',f/'records.json',f/'panel.jsonl'):
                        hashes[str(p)]=sha(p)
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
                    local.append(dict(episode_id=e['episode_id'],scene_id=scene,
                        success_replayed=success,geometry_and_primary_score_agree=True))
            finally:
                sim.close()
            frozen(dest,dict(passed=True,contract_sha256=contract_sha,
                coverage_sha256=sha(out/'exception_accounting_accepted.private.json'),
                input_hashes=hashes,records=local,steps_checked=count))
            records.extend(local);steps_checked+=count;scene_cert_hashes[str(dest)]=sha(dest)
            print('VALID_PHYSICS_REPLAY',scene,len(records),flush=True)
    require(len(records)==len(valid) and {r['episode_id'] for r in records}==valid,'Incomplete valid replay')
    require(audit(require_complete=True)==coverage,'Inputs changed during physical replay')
    result=dict(acceptance='passed',scope='All valid episodes; native exceptions remain uncompleted/unscored',
        frozen_panel_size=430,valid_episodes_replayed=len(valid),
        implementation_exceptions_uncompleted=coverage['implementation_exceptions_uncompleted'],
        pending=0,all_frozen_cases_accounted=True,scenes_with_valid_episodes=len(scene_cert_hashes),
        independent_physics_replay_complete=True,trace_steps_checked=steps_checked,
        successes=sum(r['success_replayed'] for r in records),contract_sha256=contract_sha,
        coverage_sha256=sha(out/'exception_accounting_accepted.private.json'),
        kernel_gate_sha256=sha(ROOT/'results/exception_replay_kernel_gate.json'),
        private_scene_certificate_hashes=scene_cert_hashes,
        no_exception_counted_as_navigation_failure=True)
    frozen(out/'physics_replay_valid_accepted.private.json',result)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('private_')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--kernel-gate-only',action='store_true')
    args=parser.parse_args()
    if args.kernel_gate_only:
        print(json.dumps(kernel_gate()),flush=True)
    else:
        replay()
