"""Reproduce every nonstationary native turn without policy calls or changing physics."""
import collections
import fcntl
import gzip
import hashlib
import inspect
import json
from pathlib import Path
import numpy as np
from paper_freeze import ROOT, CHAIN, DATA, frozen, sha


def main():
    import habitat
    import habitat_sim
    from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1
    from habitat.config.default import get_config
    from omegaconf import OmegaConf
    out = ROOT/'results/official_gibson_fgprompt_4901'
    inventory_path = ROOT/'results/repair_20261019/official_trace_inventory.private.json'
    inventory = json.loads(inventory_path.read_text())
    assert not inventory['metric_reconciliation_issues']
    assert sha(out/'configuration.json') == inventory['original_configuration_sha256']
    assert sha(out/'records.json') == inventory['original_records_sha256']
    for name, digest in inventory['preserved_file_hashes'].items():
        assert sha(Path(name)) == digest
    events = inventory['nonforward_motion']
    cases = collections.defaultdict(set)
    for event in events:
        assert event['action'] in (2,3), 'Do not certify unexplained STOP translation'
        cases[event['scene']].add(event['episode'])
    manifest = json.loads((ROOT/'manifests/official_gibson_imagegoal.json').read_text())
    protocol = json.loads((out/'configuration.json').read_text())
    # Frozen JSON canonicalizes mapping keys. Habitat measures depend on YAML
    # insertion order (distance -> success -> SPL), not on JSON alphabetical order.
    # Restore only the native base-config order, with every saved value unchanged.
    base = OmegaConf.to_container(get_config('benchmark/nav/imagenav/imagenav_gibson.yaml'),resolve=True)
    def ordered(value, reference):
        if not isinstance(value,dict):
            return value
        reference = reference if isinstance(reference,dict) else {}
        keys = [k for k in reference if k in value]+[k for k in value if k not in reference]
        return {k:ordered(value[k],reference.get(k)) for k in keys}
    config = OmegaConf.create(ordered(protocol['resolved_config'],base))
    config_before = OmegaConf.to_container(config,resolve=True)
    assert config_before == protocol['resolved_config']
    certified = []; replayed = []; total = 0
    controls = habitat_sim.agent.Agent.__init__.__globals__['ObjectControls']
    sources = {'raw_agent_act': inspect.getsource(habitat_sim.agent.Agent.act),
        'object_controls_action':inspect.getsource(controls.action)}
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for file in manifest['files']:
            scene = file['scene']
            if scene not in cases:
                continue
            source = json.loads(gzip.decompress(Path(file['path']).read_bytes()))
            source = source if isinstance(source,list) else source['episodes']
            selected = [e for e in source if str(e['episode_id']) in cases[scene]]
            # Include an unaffected source case as a negative control, not selected by score.
            control = next(e for e in source if str(e['episode_id']) not in cases[scene])
            selected = selected+[control]
            dataset = PointNavDatasetV1()
            dataset.from_json(json.dumps({'episodes':selected}),scenes_dir=str(DATA/'raw/scene_datasets'))
            with habitat.Env(config=config,dataset=dataset) as env:
                for _ in selected:
                    env.reset(); eid = str(env.current_episode.episode_id)
                    folder = out/'episodes'/scene/eid
                    steps = [json.loads(x) for x in (folder/'steps.jsonl').read_text().splitlines()]
                    maximum_error = 0.
                    for i, s in enumerate(steps):
                        before = np.asarray(env.sim.get_agent_state().position,np.float32).copy()
                        before_error = float(np.linalg.norm(before-np.asarray(s['before'],np.float32)))
                        assert before_error < 1e-5, (scene,eid,i,'before',before_error)
                        projected = np.asarray(env.sim.pathfinder.try_step(before,before),np.float32)
                        env.step(s['action'])
                        after = np.asarray(env.sim.get_agent_state().position,np.float32).copy()
                        error = float(np.linalg.norm(after-np.asarray(s['after'],np.float32)))
                        maximum_error = max(maximum_error,error,before_error)
                        assert error < 1e-5, (scene,eid,i,'after',error)
                        matching = [e for e in events if (e['scene'],e['episode'],e['step']) == (scene,eid,i)]
                        if matching:
                            assert len(matching) == 1
                            projection_error = float(np.linalg.norm(projected-after))
                            assert projection_error < 1e-5, 'Native zero-displacement navmesh filter does not explain turn'
                            certified.append(dict(matching[0], replay_after_error_m=error,
                                zero_displacement_native_filter_error_m=projection_error))
                        metrics = env.get_metrics()
                        assert abs(float(metrics['distance_to_goal'])-s['metrics']['distance_to_goal']) < 1e-5
                        assert float(metrics['success']) == s['metrics']['success']
                        assert abs(float(metrics['spl'])-s['metrics']['spl']) < 1e-6
                        assert bool(env.episode_over) == (i+1 == len(steps))
                    total += len(steps)
                    replayed.append(dict(scene=scene,episode=eid,steps=len(steps),max_pose_error_m=maximum_error,
                        trace_sha256=sha(folder/'steps.jsonl'),original_record_sha256=sha(folder/'record.json')))
                    print('REPLAYED',scene,eid,len(steps),maximum_error,flush=True)
    assert len(certified) == len(events)
    # Habitat Env itself binds scene/start-state fields when resetting an episode.
    # Do not confuse that native lifecycle behavior with editing saved settings.
    # Every saved value was checked before Env construction; original file bytes,
    # every replayed pose and every native metric remain independently checked.
    assert sha(out/'configuration.json') == inventory['original_configuration_sha256']
    result = dict(passed=True,inventory_sha256=sha(inventory_path),
        configuration_sha256=sha(out/'configuration.json'),records_sha256=sha(out/'records.json'),
        certified_events=certified,replayed_cases=replayed,replayed_steps=total,
        pose_tolerance_m=1e-5,original_pose_tolerance_unchanged=True,
        saved_configuration_values_identical_before_native_env=True,
        native_env_configuration_after=OmegaConf.to_container(config,resolve=True),
        metrics_and_policy_records_unchanged=True,model_calls=0,
        native_sources=sources,native_source_sha256={k:hashlib.sha256(v.encode()).hexdigest() for k,v in sources.items()},
        scope='Exact original native-configuration action replay plus native zero-displacement navmesh filter; no rescoring or model rerun')
    frozen(ROOT/'results/repair_20261019/official_motion_gate.private.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['native_sources','native_env_configuration_after']},indent=2),flush=True)


if __name__ == '__main__':
    main()
