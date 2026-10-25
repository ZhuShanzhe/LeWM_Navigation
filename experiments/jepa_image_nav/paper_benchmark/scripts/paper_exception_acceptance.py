"""Separate final gate for user-authorized original-UniGoal exception accounting.

All original common/official/preservation gates remain mandatory. Only the
impossible '430 valid UniGoal' and full-ID paired-statistics gates are replaced
by explicit complete accounting, valid-subset physics and matched-valid stats.
An accepted assessment with exceptional cases is not 1720 valid model runs.
Publication/secret scan/remote commit verification remain separate gates.
"""
import contextlib
import hashlib
import io
import json
from pathlib import Path

from paper_freeze import ROOT, sha, frozen
from paper_scope import MANIFEST, run_name
from paper_exception_audit import audit, require


def main():
    import paper_compact_acceptance as original
    with contextlib.redirect_stdout(io.StringIO()):
        original.main()
    original_path=ROOT/'results/compact430_acceptance_status.json'
    base=json.loads(original_path.read_text())
    original_names=[c['name'] for c in base['checks']]
    require(len(original_names)==19 and len(set(original_names))==19,
            'Original gate inventory changed; manually reconcile, do not silently drop a gate')
    replaced={run_name('unigoal','instance'),'Paired scene statistics'}
    require(replaced<=set(original_names),'Original gate names changed')
    checks=[dict(c) for c in base['checks'] if c['name'] not in replaced]
    coverage=audit(require_complete=False)
    checks.append(dict(name='All original UniGoal IDs accounted; exceptions unscored',
        passed=coverage['all_frozen_cases_accounted'],
        completed_valid=coverage['completed_valid'],
        implementation_exceptions_uncompleted=coverage['implementation_exceptions_uncompleted'],
        pending=coverage['pending']))
    u=ROOT/'results'/run_name('unigoal','instance')

    def check(name,path,predicate):
        if not path.exists():
            checks.append(dict(name=name,passed=False,reason='Required artifact not ready'))
            return
        data=json.loads(path.read_text())
        checks.append(dict(name=name,passed=bool(predicate(data)),sha256=sha(path),
                           artifact=str(path.relative_to(ROOT))))

    check('Independent final exception accounting',u/'exception_accounting_accepted.private.json',
        lambda d:d==coverage and d['all_frozen_cases_accounted'] and
            not d['exceptions_are_navigation_failures'] and not d['exceptions_have_valid_performance_scores'])
    check('Original replay physics/score kernel retained',ROOT/'results/exception_replay_kernel_gate.json',
        lambda d:d['passed'] and d['geometry_terminal_visibility_success_kernel_AST_identical']
            and d['original_replay_sha256']==sha(ROOT/'tools/paper_replay.py')
            and d['exception_aware_replay_sha256']==sha(ROOT/'tools/paper_exception_replay.py'))
    check('Valid-subset full physical replay',u/'physics_replay_valid_accepted.private.json',
        lambda d:d['acceptance']=='passed' and d['independent_physics_replay_complete']
            and d['valid_episodes_replayed']==coverage['completed_valid']
            and d['implementation_exceptions_uncompleted']==coverage['implementation_exceptions_uncompleted']
            and d['all_frozen_cases_accounted'] and d['pending']==0
            and d['no_exception_counted_as_navigation_failure']
            and d['coverage_sha256']==sha(u/'exception_accounting_accepted.private.json')
            and all(sha(Path(f))==h for f,h in d['private_scene_certificate_hashes'].items()))
    check('Missing-value and paired bootstrap regression',ROOT/'results/exception_stats_formula_gate.json',
        lambda d:d['passed'] and d['stats_source_sha256']==sha(ROOT/'tools/paper_exception_stats.py')
            and d['missing_cases_not_imputed_as_failure'] and d['paired_ID_mismatch_rejected']
            and d['complete_balanced_panel_bootstrap_identical'])

    def valid_stats(d):
        valid=coverage['completed_valid'];exceptions=coverage['implementation_exceptions_uncompleted']
        require(d['freeze_sha256']==sha(MANIFEST) and d['all_frozen_cases_accounted'],'Stats freeze mismatch')
        require(d['valid_completed_model_runs']==1290+valid and
                d['native_implementation_exceptions_uncompleted']==exceptions,'Stats totals mismatch')
        require(d['no_missing_or_exceptional_case_counted_as_navigation_failure'],'Exception scored as failure')
        require(d['coverage_sha256']==sha(u/'exception_accounting_accepted.private.json') and
                d['replay_sha256']==sha(u/'physics_replay_valid_accepted.private.json'), 'Stats evidence changed')
        models={x['model']:x for x in d['models']}
        require(set(models)=={'fgprompt','debit','goat','unigoal'},'Missing model statistics')
        require(all(models[m]['valid_episodes']==430 for m in ('fgprompt','debit','goat')) and
                models['unigoal']['valid_episodes']==valid,'Wrong conditional denominator')
        pairs={x['task']:x for x in d['paired_comparisons']}
        require(set(pairs)=={'place','instance'} and pairs['place']['common_valid_episodes']==430,
                'Wrong task-pair accounting')
        digest=hashlib.sha256(('\n'.join(sorted(coverage['episode_ids']['valid']))+'\n').encode()).hexdigest()
        require(pairs['instance']['common_valid_episodes']==valid and
                pairs['instance']['common_valid_ids_sha256']==digest and
                pairs['instance']['equal_input_modalities'] is False,'Unmatched or undisclosed instance comparison')
        return True

    check('Coverage-explicit common-valid paired statistics',ROOT/'results/compact430_exception_statistics.json',valid_stats)
    check('Statistics source evidence preserved',ROOT/'results/compact430_exception_statistics_input_hashes.private.json',
        lambda d:bool(d) and all(sha(Path(f))==h for f,h in d.items()))
    result=dict(assessment_accepted_under_native_exception_accounting=all(c['passed'] for c in checks),
        original_assigned_model_cases=1720,valid_completed_model_runs=1290+coverage['completed_valid'],
        native_implementation_exceptions_uncompleted=coverage['implementation_exceptions_uncompleted'],
        pending=coverage['pending'],all_1720_are_valid_navigation_scores=False,
        no_exception_counted_as_navigation_failure=True,
        official_original_gibson_episodes=4901,checks=checks,
        original_acceptance_status_retained=True,
        original_status_sha256=sha(original_path),
        replaced_gate_names=sorted(replaced),
        replacement_authority='User chose native UniGoal and separate exceptional unfinished cases,2026-10-19',
        private_assessment_source_hashes={f:sha(ROOT/'tools'/f) for f in
            ('paper_exception_audit.py','paper_exception_replay.py','paper_exception_stats.py',
             'paper_exception_acceptance.py','paper_compact_acceptance.py')},
        publication_secret_scan_and_remote_commit_verification_still_required=True)
    path=ROOT/'results/compact430_exception_acceptance_status.private.json'
    path.write_text(json.dumps(result,indent=2))
    if result['assessment_accepted_under_native_exception_accounting']:
        frozen(ROOT/'results/compact430_exception_assessment_accepted.private.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('checks','private_assessment_source_hashes')},indent=2))


if __name__=='__main__':
    main()
