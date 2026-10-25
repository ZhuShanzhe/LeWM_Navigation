"""Original Gibson ImageGoal episodes and native Habitat task/metrics, FGPrompt policy.

Runtime uses the installed Habitat0.3.1 on5090; this is explicitly not bitwise
reproduction of the author's Habitat0.2.2 environment. No source episodes resampled.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from paper_freeze import ROOT, DATA, CHAIN, sha, frozen
from paper_eval import RPC


def native_metrics(env):
    # Retain native scalar scores, not visualization-map arrays or simulator state.
    metrics = env.get_metrics()
    return {name: float(metrics[name]) for name in ['distance_to_goal', 'success', 'spl']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gate', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    author_gate = json.loads((ROOT/'results/fgprompt_author_defaults_gate.json').read_text())
    assert author_gate['passed'], 'Author-pinned geometry/camera defaults must be audited before native execution'
    assert os.environ.get('PYTHONHASHSEED') == '0', 'Pin native uniform goal-camera hashing'
    import habitat
    import habitat_sim
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    from habitat.tasks.nav.nav import ImageGoalSensor
    from habitat.config.default_structured_configs import HabitatSimRGBSensorConfig
    from habitat.config.default import get_config
    from omegaconf import OmegaConf
    from habitat_sim.utils.common import quat_to_coeffs
    manifest = json.loads((ROOT/'manifests/official_gibson_imagegoal.json').read_text())
    assert manifest['episodes'] == 4901 and len(manifest['files']) == 14
    run = 'official_gibson_fgprompt'+('_gate' if args.gate else '_smoke' if args.smoke else '_4901')
    out = ROOT/'results'/run; out.mkdir(parents=True, exist_ok=True)
    config = get_config('benchmark/nav/imagenav/imagenav_gibson.yaml')
    with habitat.config.read_write(config):
        config.habitat.seed = 0
        config.habitat.environment.max_episode_steps = 20 if args.smoke else 500
        config.habitat.environment.iterator_options.shuffle = False
        config.habitat.environment.iterator_options.cycle = False
        sim = config.habitat.simulator
        sim.forward_step_size = .25; sim.turn_angle = 30
        sim.habitat_sim_v0.allow_sliding = True
        agent = sim.agents.main_agent
        agent.height = 1.5; agent.radius = .1
        agent.sim_sensors = {'rgb_sensor': HabitatSimRGBSensorConfig(
            width=128, height=128, hfov=90, position=[0., 1.25, 0.])}
        config.habitat.task.measurements.success.success_distance = 1.
    model_root = Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
    policy_command = [str(model_root/'policy_env/bin/python'), str(model_root/'tools/baseline_policy.py'), '--model', 'fgprompt']
    source_repo = model_root/'repos/FGPrompt'
    protocol = {'task': 'Original Gibson ImageGoal validation', 'episodes': 4901,
        'official_manifest_sha256': sha(ROOT/'manifests/official_gibson_imagegoal.json'),
        'habitat_lab': habitat.__version__, 'habitat_sim': habitat_sim.__version__,
        'author_runtime': 'Habitat-sim0.2.2 and pinned author Habitat-Lab commit1f7cfbd; installed0.3.1 is a runtime adaptation',
        'resolved_config': OmegaConf.to_container(config, resolve=True),
        'author_config_files_sha256': {str(p.relative_to(source_repo)): sha(p) for p in [
            source_repo/'exp_config/base_task_config/imagenav_gibson.yaml',
            source_repo/'src/config/policy.yaml', source_repo/'src/config/sensors.yaml', source_repo/'src/config/eval.yaml']},
        'source_data_unchanged': True, 'goal_sensor': 'Native ImageGoalSensor uniform abs(hash(episode_id)) seeded rotation; no goal IDs cast',
        'author_defaults_gate_sha256': sha(ROOT/'results/fgprompt_author_defaults_gate.json'),
        'python_hash_seed': 0, 'success_measure': 'Native Habitat Success with explicit STOP and distance<1m',
        'SPL_measure': 'Native Habitat SPL', 'policy_goal_type': 'imagegoal_sensor_v2 alias only, RGB tensors unchanged',
        'official_hidden_leaderboard_score': False}
    frozen(out/'configuration.json', protocol)
    records = []; policy = None
    with (out/'batch.lock').open('a') as batch, (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(batch, fcntl.LOCK_EX|fcntl.LOCK_NB); fcntl.flock(lock, fcntl.LOCK_EX)
        sys.path.insert(0, str(CHAIN/'tools')); from chain_gpu_idle import wait
        wait()
        if not args.gate:
            gate = json.loads((ROOT/'results/official_gibson_fgprompt_gate/accepted_summary.json').read_text())
            assert gate['acceptance'] == 'passed'
            policy = RPC(policy_command, out/'policy_stderr.log', dict(os.environ,
                LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
            meta = policy.call(); assert meta['ready'] and meta['strict_state_dict']
            frozen(out/'policy.json', meta)
        try:
            with tempfile.TemporaryDirectory(prefix='official_gibson_', dir='/dev/shm') as temporary:
                temp = Path(temporary); rgb = temp/'rgb.npy'; goal = temp/'goal.npy'
                for file in manifest['files']:
                    path = Path(file['path']); assert sha(path) == file['sha256']
                    source = json.loads(gzip.decompress(path.read_bytes()))
                    source = source if isinstance(source, list) else source['episodes']
                    if args.gate or args.smoke:
                        source = source[:1]
                    dataset = PointNavDatasetV1()
                    dataset.from_json(json.dumps({'episodes': source}), scenes_dir=str(DATA/'raw/scene_datasets'))
                    with habitat.Env(config=config, dataset=dataset) as env:
                        for _ in source:
                            observations = env.reset(); e = env.current_episode
                            assert observations['rgb'].shape == (128, 128, 3)
                            assert observations['imagegoal'].shape == (128, 128, 3)
                            sid = file['scene']; eid = str(e.episode_id)
                            folder = out/'episodes'/sid/eid; folder.mkdir(parents=True, exist_ok=True)
                            completed = folder/'record.json'
                            if completed.exists():
                                records.append(json.loads(completed.read_text())); continue
                            if args.gate:
                                class FakeSim:
                                    def get_observations_at(self, position, rotation):
                                        return {'rgb': {'position': position, 'rotation': rotation}}
                                witness = SimpleNamespace(_sim=FakeSim(), _rgb_sensor_uuid='rgb')
                                actual = ImageGoalSensor._get_pointnav_episode_image_goal(witness, e)
                                rng = np.random.RandomState(abs(hash(e.episode_id)) % (2**32))
                                angle = rng.uniform(0, 2*np.pi)
                                expected = [0, np.sin(angle/2), 0, np.cos(angle/2)]
                                assert np.allclose(actual['rotation'], expected, atol=0, rtol=0)
                                assert actual['position'] == np.asarray(e.goals[0].position, np.float32).tolist()
                                before = env.sim.get_agent_state()
                                q0 = quat_to_coeffs(before.rotation)
                                env.step(2)
                                after = env.sim.get_agent_state(); q1 = quat_to_coeffs(after.rotation)
                                assert np.linalg.norm(np.asarray(after.position)-np.asarray(before.position)) < 1e-5
                                turn = 2*math.acos(float(np.clip(abs(np.dot(q0, q1)), -1, 1)))
                                assert abs(turn-math.pi/6) < 1e-4
                                env.step(0)
                                metrics = env.get_metrics()
                                assert 'success' in metrics and 'spl' in metrics and env.episode_over
                                r = {'scene_id': sid, 'episode_id': eid, 'passed': True,
                                    'native_episode_id_type': type(e.episode_id).__name__, 'policy_performance': False}
                            else:
                                from paper_archive import archive
                                archive(folder, ['steps.jsonl', 'runtime_failure.json'])
                                np.save(goal, observations['imagegoal'])
                                policy.call({'op': 'reset', 'goal': str(goal)})
                                initial_metrics = native_metrics(env)
                                with (folder/'steps.jsonl').open('x') as trace:
                                    count = 0
                                    while not env.episode_over:
                                        np.save(rgb, observations['rgb'])
                                        decision = policy.call({'op': 'step', 'rgb': str(rgb)})
                                        action = int(decision['action']); assert action in range(4)
                                        before = env.sim.get_agent_state()
                                        observations = env.step(action)
                                        after = env.sim.get_agent_state()
                                        trace.write(json.dumps({'t': count, 'action': action,
                                            'logits': decision['logits'], 'before': before.position,
                                            'after': after.position, 'metrics': native_metrics(env)},
                                            default=lambda a: np.asarray(a).tolist())+'\n'); trace.flush(); count += 1
                                r = {'scene_id': sid, 'episode_id': eid, 'status': 'complete',
                                    'steps': count, 'final_action': action, 'initial_native_metrics': initial_metrics,
                                    'source_geodesic_distance_m': e.info.get('geodesic_distance'),
                                    'native_metrics': native_metrics(env)}
                            frozen(completed, r); records.append(r)
                            print(json.dumps({'run': run, 'completed': len(records), 'scene': sid}), flush=True)
                    if args.gate or args.smoke:
                        break
        finally:
            if policy is not None:
                policy.close()
    if args.gate:
        frozen(out/'accepted_summary.json', {'acceptance': 'passed', 'episodes': len(records),
            'native_goal_rotation_regression': True, 'native30deg_turn_checked': True,
            'native_STOP_and_metrics_checked': True, 'model_calls': 0,
            'scope': 'One native interface episode, not a paper-performance reproduction'})
    else:
        frozen(out/'records.json', records)
        frozen(out/'execution_complete.json', {'complete': len(records) == (1 if args.smoke else 4901),
            'episodes': len(records), 'independent_acceptance_pending': True})


if __name__ == '__main__':
    main()
