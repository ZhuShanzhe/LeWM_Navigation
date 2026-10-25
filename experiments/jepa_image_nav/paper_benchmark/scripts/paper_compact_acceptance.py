"""Final compact-scope evidence gate; no obsolete full legacy-quota requirement."""
import json
from pathlib import Path
from paper_freeze import ROOT, CHAIN, sha, frozen
from paper_scope import COUNT, MANIFEST, TRACKS, run_name


def main():
    checks = []
    def check(name, path, predicate):
        if not path.exists():
            checks.append({'name': name, 'passed': False, 'reason': 'Required artifact not ready'})
            return
        data = json.loads(path.read_text())
        checks.append({'name': name, 'passed': bool(predicate(data)), 'sha256': sha(path),
            'artifact': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)})
    check('Compact freeze', MANIFEST, lambda d: d['formal_model_runs'] == 1720)
    check('Compact CPU checks', ROOT/'results/compact430_cpu_gate.json', lambda d: d['passed'])
    check('Selected geometry', ROOT/'results/geometry_compact430_gate.json', lambda d:
        d['passed'] and d['episodes_checked'] == COUNT*2 and d['freeze_sha256'] == sha(MANIFEST))
    check('Native input interface smokes', ROOT/'results/compact430_smoke_gate.json', lambda d: d['passed'])
    check('Native collision interface regression', ROOT/'results/repair_20261019/collision_interface_gate.json',
        lambda d: d['passed'] and d['model_calls'] == 0 and d['sim_sha256'] == sha(ROOT/'tools/paper_sim.py'))
    check('Repair source and protected-result integrity', ROOT/'results/repair_20261019/repair_integrity_gate.json',
        lambda d: d['passed'] and d['AST_difference_only_native_collision_read_and_type_check']
            and d['repaired_sim_sha256'] == sha(ROOT/'tools/paper_sim.py'))
    check('UniGoal base-config guard compatibility', ROOT/'results/repair_20261019/unigoal_yaml_guard_gate.json',
        lambda d: d['passed'] and d['all_runtime_configuration_values_identical']
            and d['adapter_sha256'] == sha(ROOT/'tools/paper_unigoal.py'))
    check('Original native turn projection replay', ROOT/'results/repair_20261019/official_motion_gate.private.json',
        lambda d: d['passed'] and d['original_pose_tolerance_unchanged'] and d['metrics_and_policy_records_unchanged'])
    check('UniGoal actual loader equivalence', ROOT/'results/unigoal_actual_model_load_gate.json', lambda d:
        d['passed'] and d['bit_exact_model_states_and_fixed_image_outputs'] and d['CPU_and_CUDA_RNG_states_equal'])
    check('UniGoal non-mutating call diagnostics', ROOT/'results/unigoal_call_diagnostics_gate.json', lambda d:
        d['passed'] and d['requests_byte_identical_to_native'] and d['native_return_values_identical'])
    from paper_runtime import ARTIFACT, sources
    check('Frozen execution sources', ARTIFACT, lambda d:
        d['freeze_sha256'] == sha(MANIFEST) and d['sources'] == sources())
    for model, task in TRACKS+[('unigoal', 'instance')]:
        check(run_name(model, task), ROOT/'results'/run_name(model, task)/'physics_replay_accepted.json',
            lambda d: d['acceptance'] == 'passed' and d['episodes'] == COUNT and d['scenes'] == 43
                and d['independent_physics_replay_complete'])
    check('Paired scene statistics', ROOT/'results/compact430_statistics.json', lambda d:
        d['freeze_sha256'] == sha(MANIFEST) and len(d['models']) == 4 and len(d['paired_comparisons']) == 2)
    check('Original Gibson official image-goal anchor', ROOT/'results/official_gibson_fgprompt_4901/accepted_summary.json',
        lambda d: d['acceptance'] == 'passed' and d['episodes'] == 4901 and d['original_episode_identity_coverage_verified'])
    check('Safe legacy scope boundary', ROOT/'results/compact430_legacy_pause.json', lambda d:
        d['status'] == 'paused_by_user_scope_change' and d['current_episode_finished']
            and not d['original_records_and_logs_deleted'] and not d['unfinished_counted_as_failure'])
    preserve = json.loads((ROOT/'results/compact430_legacy_pause_requested.json').read_text())
    unchanged = all(sha(Path(p)) == h for p, h in preserve['protected_completed_files'].items())
    checks.append({'name': 'Original completed results byte-preserved', 'passed': unchanged,
        'files': len(preserve['protected_completed_files'])})
    result = {'all_compact_evaluations_accepted': all(c['passed'] for c in checks), 'checks': checks,
        'formal_controlled_model_runs': 1720, 'official_original_gibson_episodes': 4901,
        'legacy_incomplete_quotas_not_required': True, 'missing_official_instance_scene_license_disclosed': True,
        'publication_and_remote_commit_verification_are_separate_required_gates': True}
    # Mutable status is not a completion certificate; the latter is immutable and only written on success.
    (ROOT/'results/compact430_acceptance_status.json').write_text(json.dumps(result, indent=2))
    if result['all_compact_evaluations_accepted']:
        frozen(ROOT/'results/compact430_all_accepted.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
