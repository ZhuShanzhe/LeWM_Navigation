"""Read-only evidence accounting for native UniGoal exceptions; never score a crash.

User selected the unchanged native implementation on 2026-10-19. This auditor
does not execute policies, retry episodes, alter traces, or replace old gates.
Partial coverage is a progress audit, never a final evaluation certificate.
"""
import argparse
import collections
import json
import re
from pathlib import Path

from paper_freeze import ROOT, sha, frozen
from paper_runtime import verify
from paper_scope import COUNT, MANIFEST, panel_rows, run_name

NATIVE_STACK = ('src/graph/graph.py', 'in get_goal',
                'src/utils/fmm/fmm_planner.py', 'in set_goal',
                'self.traversible[goal_x, goal_y]')


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def classify(accepted_exists, exception_exists):
    require(not (accepted_exists and exception_exists), 'Overlapping outcome classes')
    return 'valid' if accepted_exists else ('exception' if exception_exists else 'pending')


def validate_exception(certificate, episode, folder, log, runtime_path):
    require(certificate['episode_id'] == episode['episode_id'], 'Wrong exception episode')
    require(certificate['scene_id'] == episode['scene_id'], 'Wrong exception scene')
    require(certificate['status'] == 'native_implementation_exception_uncompleted',
            'Unrecognized exception status')
    for key in ('performance_score_valid', 'counted_as_navigation_failure'):
        require(certificate[key] is False, 'Exception cannot carry a valid score/failure')
    for key in ('original_native_framework_preserved', 'no_synthetic_STOP_or_score'):
        require(certificate[key] is True, 'Missing native-preservation assertion')
    require(certificate['frozen_runtime_sha256'] == sha(runtime_path), 'Runtime certificate changed')
    require(certificate['episode_panel_sha256'] == sha(folder/'panel.jsonl'), 'Attempt panel changed')
    require(json.loads((folder/'panel.jsonl').read_text()) == episode, 'Non-frozen attempted episode')
    failure_path = folder/'runtime_failure.json'
    failure = json.loads(failure_path.read_text())
    require(certificate['exception'] == failure, 'Failure evidence changed')
    require(failure['error'] == 'IndexError' and re.fullmatch(
        r'index -?\d+ is out of bounds for axis [01] with size \d+', failure['detail']),
        'Only precisely recognized native map-bound errors may be excluded')
    require(certificate['evidence_log_sha256'] == sha(log), 'Exception log changed')
    tail = log.read_text(errors='replace')[-12000:]
    require(all(s in tail for s in NATIVE_STACK), 'Native Graph/FMM call chain missing')
    protected = certificate['protected_attempt_file_hashes']
    paths = {Path(p) for p in protected}
    require(paths == {p for p in folder.iterdir() if p.is_file()}, 'Attempt file inventory changed')
    require(failure_path in paths and folder/'panel.jsonl' in paths, 'Missing protected evidence')
    require(all(p.parent == folder and sha(p) == protected[str(p)] for p in paths),
            'Private attempt evidence changed or escapes episode folder')
    require(not (folder/'accepted_summary.json').exists(), 'Exception has accepted navigation score')


def audit(require_complete=False):
    verify()
    rows = panel_rows('instance')
    run = ROOT/'results'/run_name('unigoal', 'instance')
    episodes = run/'episodes'
    known = {e['episode_id'] for e in rows}
    require(len(known) == COUNT, 'Frozen ID inventory incomplete')
    require({p.name for p in episodes.iterdir() if p.is_dir()} <= known, 'Unexpected episode folder')
    certdir = run/'implementation_exceptions'
    require({p.stem for p in certdir.glob('*.json')} <= known, 'Unexpected exception certificate')
    contract = sha(ROOT/'protocol/common_v1.json')
    runtime = ROOT/'manifests/compact430_runtime_sources.json'
    groups = {'valid': [], 'exception': [], 'pending': []}
    scenes = collections.defaultdict(collections.Counter)
    inputs = {str(MANIFEST): sha(MANIFEST), str(runtime): sha(runtime)}
    for e in rows:
        eid = e['episode_id']; folder = episodes/eid
        accepted = folder/'accepted_summary.json'; cert = certdir/(eid+'.json')
        kind = classify(accepted.exists(), cert.exists())
        groups[kind].append(eid); scenes[e['scene_id']][kind] += 1
        if kind == 'exception':
            log = ROOT/'logs'/(run.name+'_'+eid+'.log')
            validate_exception(json.loads(cert.read_text()), e, folder, log, runtime)
            inputs[str(cert)] = sha(cert)
        elif kind == 'valid':
            a = json.loads(accepted.read_text())
            require(a['acceptance'] == 'passed' and a['episodes'] == 1, 'Invalid episode acceptance')
            require(a['contract_sha256'] == contract, 'Changed physics/scoring contract')
            require(not (folder/'runtime_failure.json').exists(), 'Accepted episode has runtime failure')
            require(a['episode_panel_sha256'] == sha(folder/'panel.jsonl'), 'Accepted panel changed')
            require(json.loads((folder/'panel.jsonl').read_text()) == e, 'Accepted ID substitution')
            require(a['trace_sha256'] == sha(folder/(eid+'.steps.jsonl')), 'Accepted trace changed')
            require(a['records'] == json.loads((folder/'records.json').read_text()), 'Accepted records changed')
            require(len(a['records']) == 1 and a['records'][0]['status'] == 'complete'
                    and a['records'][0]['episode_id'] == eid, 'Wrong completed record')
            inputs[str(accepted)] = sha(accepted)
    require(len(scenes) == 43 and all(sum(c.values()) == 10 for c in scenes.values()),
            'Frozen full scene coverage changed')
    complete = not groups['pending']
    require(not require_complete or complete, 'All original 430 IDs are not yet accounted')
    if complete:
        execution = json.loads((run/'native_execution_accounted.json').read_text())
        require(execution['all_frozen_cases_accounted'] is True, 'Scheduler not finished')
        require(execution['completed_valid'] == len(groups['valid']) and
                execution['implementation_exceptions_uncompleted'] == len(groups['exception']),
                'Scheduler/auditor counts differ')
        inputs[str(run/'native_execution_accounted.json')] = sha(run/'native_execution_accounted.json')
    return dict(audit_passed=True, frozen_panel_size=COUNT, completed_valid=len(groups['valid']),
        implementation_exceptions_uncompleted=len(groups['exception']), pending=len(groups['pending']),
        all_frozen_cases_accounted=complete, frozen_panel_sha256=sha(MANIFEST),
        runtime_manifest_sha256=sha(runtime), contract_sha256=contract,
        episode_ids=groups, per_scene={s: dict(c) for s,c in sorted(scenes.items())},
        exceptions_are_navigation_failures=False, exceptions_have_valid_performance_scores=False,
        performance_denominator='Valid completed episodes only; report full-panel coverage separately',
        paired_comparison='GOAT and UniGoal on the identical intersection of valid episode IDs',
        physical_replay_statistics_and_publication_still_separate=True,
        private_evidence_hashes=inputs)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = audit(args.require_complete)
    output = ROOT/'results'/run_name('unigoal','instance')
    if args.require_complete:
        frozen(output/'exception_accounting_accepted.private.json', result)
    else:
        (output/'exception_accounting_progress.private.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items()
        if k not in ('episode_ids','private_evidence_hashes','per_scene')}), flush=True)


if __name__ == '__main__':
    main()
