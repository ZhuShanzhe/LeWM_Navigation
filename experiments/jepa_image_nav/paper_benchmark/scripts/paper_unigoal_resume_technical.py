"""Preserve native UniGoal; quarantine narrowly evidenced map-bound exceptions, unscored.

User explicitly selected preserve-original/separate-exceptions on 2026-10-19.
The original 430-case panel and every runtime source remain unchanged. This is
scheduling and accounting only; no clipping, enlarged map, fallback or fake STOP.
"""
import fcntl
import json
import re
import sys
import subprocess
import time
from pathlib import Path
from paper_freeze import ROOT, CHAIN, DATA, frozen, sha
from paper_scope import panel_rows, run_name
from paper_runtime import verify
from paper_queue import execute


RUN = ROOT/'results'/run_name('unigoal','instance')
EXCEPTIONS = RUN/'implementation_exceptions'


def quarantine(folder, episode):
    """Only the observed native Graph.get_goal -> FMM.set_goal array-bound error."""
    failure = folder/'runtime_failure.json'
    log = ROOT/'logs'/(RUN.name+'_'+episode['episode_id']+'.log')
    if not failure.exists() or not log.exists():
        return False
    value = json.loads(failure.read_text())
    if value.get('error') != 'IndexError' or not re.fullmatch(
            r'index -?\d+ is out of bounds for axis [01] with size \d+',value.get('detail','')):
        return False
    text = log.read_text(errors='replace')
    tail = text[-12000:]
    if not all(s in tail for s in ['src/graph/graph.py','in get_goal',
                                  'src/utils/fmm/fmm_planner.py','in set_goal',
                                  "self.traversible[goal_x, goal_y]"]):
        return False
    protected = {str(p):sha(p) for p in folder.iterdir() if p.is_file()}
    certificate = dict(episode_id=episode['episode_id'],scene_id=episode['scene_id'],
        status='native_implementation_exception_uncompleted',performance_score_valid=False,
        counted_as_navigation_failure=False,exception=value,
        original_native_framework_preserved=True,no_synthetic_STOP_or_score=True,
        authorization='User chose preserve original and separate exceptional episodes,2026-10-19',
        frozen_runtime_sha256=sha(ROOT/'manifests/compact430_runtime_sources.json'),
        episode_panel_sha256=sha(folder/'panel.jsonl'),protected_attempt_file_hashes=protected,
        evidence_log_sha256=sha(log))
    EXCEPTIONS.mkdir(parents=True,exist_ok=True)
    frozen(EXCEPTIONS/(episode['episode_id']+'.json'),certificate)
    print('QUARANTINED_UNSCORED',episode['episode_id'],value['detail'],flush=True)
    return True


def accounting(rows):
    completed = []; exceptions = []; pending = []
    for episode in rows:
        eid = episode['episode_id']; folder = RUN/'episodes'/eid
        accepted = folder/'accepted_summary.json'; exceptional = EXCEPTIONS/(eid+'.json')
        assert not (accepted.exists() and exceptional.exists())
        if accepted.exists():
            assert json.loads(accepted.read_text())['acceptance'] == 'passed'
            completed.append(eid)
        elif exceptional.exists():
            value = json.loads(exceptional.read_text())
            assert value['status'] == 'native_implementation_exception_uncompleted'
            assert not value['counted_as_navigation_failure']
            assert all(sha(Path(p)) == h for p,h in value['protected_attempt_file_hashes'].items())
            assert value['evidence_log_sha256'] == sha(ROOT/'logs'/(RUN.name+'_'+eid+'.log'))
            exceptions.append(eid)
        else:
            pending.append(eid)
    state = dict(frozen_panel_size=len(rows),completed_valid=len(completed),
        implementation_exceptions_uncompleted=len(exceptions),pending=len(pending),
        all_frozen_cases_accounted=len(completed)+len(exceptions)==len(rows),
        all_430_valid_episodes_completed=len(completed)==len(rows),
        exception_episode_ids=exceptions,exception_counted_as_navigation_failure=False,
        native_model_and_frozen_protocol_unchanged=True,updated_unix_s=time.time())
    (RUN/'coverage_status.json').write_text(json.dumps(state,indent=2))
    return state


def main():
    with (ROOT/'unigoal_technical_resume.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        verify();rows=panel_rows('instance')
        prefix=ROOT/'results/repair_20261019/unigoal_resume_accepted_prefix.json'
        protected=(json.loads(prefix.read_text()) if prefix.exists() else
            {str(p):sha(p) for p in (RUN/'episodes').glob('*/accepted_summary.json')})
        assert all(sha(Path(p))==h for p,h in protected.items())
        frozen(prefix,protected)
        for episode in rows:
            folder=RUN/'episodes'/episode['episode_id'];folder.mkdir(parents=True,exist_ok=True)
            if (folder/'accepted_summary.json').exists():
                continue
            if (EXCEPTIONS/(episode['episode_id']+'.json')).exists():
                continue
            if (folder/'runtime_failure.json').exists():
                assert quarantine(folder,episode),'Unknown existing failure; do not silently skip or rerun'
                accounting(rows);continue
            frozen(folder/'panel.jsonl',(json.dumps(episode,sort_keys=True)+'\n').encode())
            try:
                execute([str(CHAIN/'policy_env/bin/python'),'-u',str(ROOT/'tools/paper_unigoal.py'),
                    '--panel',str(folder/'panel.jsonl'),'--output',str(folder),'--max-actions','500'],
                    RUN.name+'_'+episode['episode_id'])
            except subprocess.CalledProcessError:
                if not quarantine(folder,episode):
                    raise
            state=accounting(rows)
            assert all(sha(Path(p))==h for p,h in protected.items())
            if state['completed_valid'] and state['completed_valid']%30==0:
                execute([str(CHAIN/'goat_env/bin/python'),str(ROOT/'tools/paper_eta.py')],
                        'compact430_timing_estimate')
            print('COVERAGE',json.dumps(state),flush=True)
        state=accounting(rows);assert state['all_frozen_cases_accounted']
        final=dict(state);final.pop('updated_unix_s')
        frozen(RUN/'native_execution_accounted.json',final)
        # A complete accounting certificate is NOT a claim of 430 valid completions.
        # Exception-aware physical replay/statistics are separate required gates.
        print('NATIVE_EXECUTION_ACCOUNTED; EXCEPTION_AWARE_ACCEPTANCE_STILL_REQUIRED',flush=True)


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        failure=dict(status='technical_failure',error=type(error).__name__,detail=str(error),
            counted_as_navigation_failure=False,time_unix_s=time.time())
        frozen(ROOT/'results/queue_failures'/('unigoal_resume_'+str(time.time_ns())+'.json'),failure)
        (ROOT/'results/unigoal_resume_failure.json').write_text(json.dumps(failure,indent=2))
        raise
