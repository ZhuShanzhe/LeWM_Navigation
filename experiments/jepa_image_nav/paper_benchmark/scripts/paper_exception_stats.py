"""Coverage-explicit, scene-cluster statistics with common-valid-ID paired tests.

This does not replace original immutable statistics/gates. A native crash has
no success/SPL value: no zero filling, substitute episode, or failure imputation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from paper_freeze import ROOT, sha, frozen
from paper_scope import COUNT, MANIFEST, panel_rows, run_name
from paper_common import CONTRACT
from paper_exception_audit import audit, require

SEED=20261017
REPLICATES=10000


def arrays(rows,scenes):
    require(len({r['episode_id'] for r in rows})==len(rows),'Repeated episode')
    require({r['scene_id'] for r in rows}<=set(scenes),'Unexpected scene')
    n=np.asarray([sum(r['scene_id']==s for r in rows) for s in scenes],dtype=float)
    sums=np.asarray([[sum(r[k] for r in rows if r['scene_id']==s)
        for k in ('success','spl')] for s in scenes],dtype=float)
    require(n.sum()>0,'No valid performance observations')
    return sums,n


def estimates(sums,n,draws):
    eligible=n>0
    means=np.divide(sums,n[:,None],out=np.zeros_like(sums),where=n[:,None]>0)
    point_scene=means[eligible].mean(axis=0)
    point_episode=sums.sum(axis=0)/n.sum()
    drawn_n=n[draws].sum(axis=1)
    drawn_eligible=eligible[draws].sum(axis=1)
    usable=drawn_n>0
    require(usable.any(),'No nonempty bootstrap replicate')
    boot_episode=sums[draws].sum(axis=1)[usable]/drawn_n[usable,None]
    boot_scene=means[draws].sum(axis=1)[usable]/drawn_eligible[usable,None]
    return dict(scene=point_scene,episode=point_episode,boot_scene=boot_scene,
                boot_episode=boot_episode,nonempty_replicates=int(usable.sum()))


def aggregate(rows,scenes,draws):
    sums,n=arrays(rows,scenes);est=estimates(sums,n,draws)
    return dict(valid_episodes=len(rows),scenes_with_valid_observations=int((n>0).sum()),
        frozen_scenes=len(scenes),successes=int(sums[:,0].sum()),
        scene_weighted_SR=float(est['scene'][0]),scene_weighted_SPL=float(est['scene'][1]),
        episode_weighted_SR=float(est['episode'][0]),episode_weighted_SPL=float(est['episode'][1]),
        scene_cluster_bootstrap95_SR=np.quantile(est['boot_scene'][:,0],[.025,.975]).tolist(),
        scene_cluster_bootstrap95_SPL=np.quantile(est['boot_scene'][:,1],[.025,.975]).tolist(),
        episode_weighted_cluster_bootstrap95_SR=np.quantile(est['boot_episode'][:,0],[.025,.975]).tolist(),
        episode_weighted_cluster_bootstrap95_SPL=np.quantile(est['boot_episode'][:,1],[.025,.975]).tolist(),
        valid_count_per_frozen_scene={s:int(v) for s,v in zip(scenes,n)},
        nonempty_bootstrap_replicates=est['nonempty_replicates'],
        interpretation='Conditional on valid completed episodes; no assertion that missingness is random',
        missing_scene_handling='No imputed score; macro means use scenes with valid observations')


def pair(first,second,scenes,draws):
    a={r['episode_id']:r for r in first};b={r['episode_id']:r for r in second}
    require(set(a)==set(b),'Paired statistics require exactly matching valid IDs')
    require(all(a[k]['scene_id']==b[k]['scene_id'] for k in a),'Mismatched paired scenes')
    x,n=arrays(first,scenes);y,m=arrays(second,scenes)
    require(np.array_equal(n,m),'Mismatched per-scene paired counts')
    est=estimates(y-x,n,draws)
    return dict(common_valid_episodes=len(a),
        common_valid_ids_sha256=hashlib.sha256(('\n'.join(sorted(a))+'\n').encode()).hexdigest(),
        common_valid_per_scene={s:int(v) for s,v in zip(scenes,n)},
        delta_scene_SR=float(est['scene'][0]),delta_scene_SPL=float(est['scene'][1]),
        delta_episode_SR=float(est['episode'][0]),delta_episode_SPL=float(est['episode'][1]),
        paired_scene_bootstrap95_delta_SR=np.quantile(est['boot_scene'][:,0],[.025,.975]).tolist(),
        paired_scene_bootstrap95_delta_SPL=np.quantile(est['boot_scene'][:,1],[.025,.975]).tolist(),
        first_common_valid_statistics=aggregate(first,scenes,draws),
        second_common_valid_statistics=aggregate(second,scenes,draws),
        conditional_comparison_not_an_imputed_full_panel_ranking=True)


def load(model,task,ids,inputs):
    out=ROOT/'results'/run_name(model,task);rows=[]
    for e in panel_rows(task):
        if e['episode_id'] not in ids:
            continue
        f=out/'episodes'/e['episode_id'];p=f/'accepted_summary.json'
        a=json.loads(p.read_text());r=a['records'][0]
        require(a['acceptance']=='passed' and a['episodes']==1,'Episode not accepted')
        require(a['contract_sha256']==sha(ROOT/'protocol/common_v1.json'),'Changed contract')
        require(a['episode_panel_sha256']==sha(f/'panel.jsonl') and
                json.loads((f/'panel.jsonl').read_text())==e,'Changed episode panel')
        require(a['trace_sha256']==sha(f/(e['episode_id']+'.steps.jsonl')),'Trace changed')
        require(a['records']==json.loads((f/'records.json').read_text()),'Record changed')
        require(r['status']=='complete' and r['episode_id']==e['episode_id'],'Wrong completed ID')
        require(r['success'] in (0,1) and 0<=r['spl']<=1,'Invalid performance value')
        inputs[str(p)]=sha(p)
        rows.append(dict(episode_id=e['episode_id'],scene_id=e['scene_id'],
            success=r['success'],spl=r['spl'],distance_bin=e['distance_bin'],detour_bin=e['detour_bin']))
    require({r['episode_id'] for r in rows}==ids,'Incomplete valid ID loading')
    return rows


def formula_gate():
    scenes=['a','b','c']
    rows=[dict(episode_id='a1',scene_id='a',success=1,spl=.8),
          dict(episode_id='a2',scene_id='a',success=0,spl=0.),
          dict(episode_id='b1',scene_id='b',success=1,spl=.5)]
    draws=np.random.default_rng(SEED).integers(0,3,size=(REPLICATES,3))
    result=aggregate(rows,scenes,draws)
    require(np.isclose(result['scene_weighted_SR'],.75),'Macro weighting regression')
    require(np.isclose(result['episode_weighted_SR'],2/3),'Exception was imputed as failure')
    require(result['valid_count_per_frozen_scene']==dict(a=2,b=1,c=0),'Empty-scene coverage lost')
    identical=pair(rows,rows,scenes,draws)
    require(identical['delta_scene_SR']==0 and
        identical['paired_scene_bootstrap95_delta_SPL']==[0.,0.],'Paired uncertainty regression')
    try:
        pair(rows,rows[:-1],scenes,draws)
    except AssertionError:
        pass
    else:
        raise AssertionError('Mismatched paired IDs were silently accepted')
    # Equal-size complete scenes reproduce the original arithmetic exactly.
    complete=rows+[dict(episode_id='b2',scene_id='b',success=0,spl=0.),
                   dict(episode_id='c1',scene_id='c',success=0,spl=0.),
                   dict(episode_id='c2',scene_id='c',success=1,spl=.3)]
    sums,n=arrays(complete,scenes);original=(sums/n[:,None])[draws].mean(axis=1)
    candidate=estimates(sums,n,draws)['boot_scene']
    require(np.array_equal(original,candidate),'Complete-panel bootstrap differs from original formula')
    gate=dict(passed=True,stats_source_sha256=sha(Path(__file__)),policy_calls=0,GPU_calls=0,
        missing_cases_not_imputed_as_failure=True,paired_ID_mismatch_rejected=True,
        complete_balanced_panel_bootstrap_identical=True,seed=SEED,replicates=REPLICATES)
    frozen(ROOT/'results/exception_stats_formula_gate.json',gate)
    return gate


def main():
    formula_gate();coverage=audit(require_complete=True);inputs={}
    u=ROOT/'results'/run_name('unigoal','instance')
    physics=u/'physics_replay_valid_accepted.private.json';p=json.loads(physics.read_text())
    require(p['acceptance']=='passed' and p['independent_physics_replay_complete'],
            'Valid-subset independent replay missing')
    require(p['valid_episodes_replayed']==coverage['completed_valid'] and p['pending']==0,
            'Replay coverage disagrees')
    require(p['coverage_sha256']==sha(u/'exception_accounting_accepted.private.json'),
            'Replay coverage certificate changed')
    require(json.loads((u/'exception_accounting_accepted.private.json').read_text())==coverage,
            'Accounting certificate disagrees')
    require(all(sha(Path(f))==h for f,h in p['private_scene_certificate_hashes'].items()),
            'Scene replay evidence changed')
    for filename in p['private_scene_certificate_hashes']:
        scene=json.loads(Path(filename).read_text())
        require(scene['passed'] and all(sha(Path(f))==h for f,h in scene['input_hashes'].items()),
                'Valid-subset physical replay inputs changed')
    inputs[str(physics)]=sha(physics)
    panels={t:panel_rows(t) for t in ('place','instance')}
    scenes=sorted({e['scene_id'] for e in panels['place']});require(len(scenes)==43,'Wrong scene panel')
    draws=np.random.default_rng(SEED).integers(0,len(scenes),size=(REPLICATES,len(scenes)))
    valid=set(coverage['episode_ids']['valid']);summaries=[];data={}
    for model,task in [('fgprompt','place'),('debit','place'),('goat','instance'),('unigoal','instance')]:
        out=ROOT/'results'/run_name(model,task)
        ids=valid if model=='unigoal' else {e['episode_id'] for e in panels[task]}
        if model!='unigoal':
            gate=out/'physics_replay_accepted.json';g=json.loads(gate.read_text())
            require(g['acceptance']=='passed' and g['episodes']==COUNT and
                    g['independent_physics_replay_complete'],'Full baseline physics missing')
            for filename in (out/'physics_replay').glob('*.json'):
                scene=json.loads(filename.read_text())
                require(scene['passed'] and all(sha(Path(f))==h for f,h in scene['input_hashes'].items()),
                        'Full baseline physical replay inputs changed')
                inputs[str(filename)]=sha(filename)
            inputs[str(gate)]=sha(gate)
        rows=load(model,task,ids,inputs);data[model]=rows
        summary=aggregate(rows,scenes,draws)
        summary.update(model=model,task=task,frozen_panel_size=COUNT,
            implementation_exceptions_uncompleted=COUNT-len(rows),pending=0,
            native_used_inputs=CONTRACT['policy_inputs'][dict(fgprompt='FGPrompt',debit='DEBiT',
                goat='GOAT',unigoal='UniGoal')[model]])
        summaries.append(summary)
    place=pair(data['fgprompt'],data['debit'],scenes,draws)
    place.update(task='place',difference='DEBiT minus FGPrompt',equal_input_modalities=True)
    common_goat=[r for r in data['goat'] if r['episode_id'] in valid]
    instance=pair(common_goat,data['unigoal'],scenes,draws)
    instance.update(task='instance',difference='UniGoal minus GOAT on common valid IDs',
        equal_input_modalities=False,
        interpretation='Whole-system comparison; depth, localization and native action heads differ')
    result=dict(scope='Frozen HSSD43 derived cross-domain image navigation, not official hidden test or language VLN',
        freeze_sha256=sha(MANIFEST),all_frozen_cases_accounted=True,
        original_formal_model_case_assignments=1720,
        valid_completed_model_runs=1290+coverage['completed_valid'],
        native_implementation_exceptions_uncompleted=coverage['implementation_exceptions_uncompleted'],
        no_missing_or_exceptional_case_counted_as_navigation_failure=True,
        bootstrap_seed=SEED,bootstrap_replicates=REPLICATES,unit_of_resampling='Frozen scene cluster',
        models=summaries,paired_comparisons=[place,instance],
        coverage_sha256=sha(u/'exception_accounting_accepted.private.json'),
        replay_sha256=sha(physics),stats_formula_gate_sha256=sha(ROOT/'results/exception_stats_formula_gate.json'),
        uncertainty_does_not_cover_nonrandom_missingness_training_variance_or_pretraining_exposure=True)
    frozen(ROOT/'results/compact430_exception_statistics.json',result)
    frozen(ROOT/'results/compact430_exception_statistics_input_hashes.private.json',inputs)
    print(json.dumps(dict(statistics_complete=True,models=4,paired_comparisons=2,
                         valid_runs=result['valid_completed_model_runs'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--formula-gate-only',action='store_true')
    args=parser.parse_args()
    if args.formula_gate_only:
        print(json.dumps(formula_gate()),flush=True)
    else:
        main()
