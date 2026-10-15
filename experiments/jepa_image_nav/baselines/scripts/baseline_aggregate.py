"""Aggregate complete frozen-panel runs and uncertainty by resampling scenes."""
import collections, hashlib, json, math
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')

def mean(rows,key):return float(np.mean([r[key] for r in rows])) if rows else None

def scene_ci(rows,key,seed=20261015):
    groups=collections.defaultdict(list)
    for r in rows:groups[r['scene_id']].append(r[key])
    vals=np.array([np.mean(v) for v in groups.values()]);rng=np.random.default_rng(seed)
    samples=vals[rng.integers(0,len(vals),(5000,len(vals)))].mean(axis=1)
    return np.percentile(samples,[2.5,97.5]).tolist()

def aggregate(folder):
    summaries=[json.loads(p.read_text()) for p in sorted(folder.glob('*/summary.json'))]
    if not summaries:return None
    panel=BASE/'plans'/summaries[0]['panel'];expected=[json.loads(l) for l in panel.read_text().splitlines()]
    planned={r['episode_id'] for r in expected};rows=[];missing_scenes=[]
    completed_scenes={s['scene'] for s in summaries}
    for scene in sorted(completed_scenes):
        rows.extend(json.loads(l) for l in (folder/scene/'episodes.jsonl').read_text().splitlines())
    for row in rows:
        instance=row['task']=='instance_image_nav'
        final_near=row['final_distance_m']<(.5 if instance else 1.) and (row.get('target_visible_at_stop') or not instance)
        row['reached_success_region']=bool(row['reached_success_region'] or final_near)
        if final_near and not row['success'] and not row.get('wrong_instance_stop'):
            row['outcome_tag']='reached_success_region_but_failed_to_stop'
    ids=[r['episode_id'] for r in rows]
    assert len(ids)==len(set(ids)) and not(set(ids)-planned)
    complete=set(ids)==planned
    summary={'run':folder.name,'model':summaries[0]['model'],'panel':summaries[0]['panel'],
             'panel_sha256':hashlib.sha256(panel.read_bytes()).hexdigest(),'complete_frozen_panel':complete,
             'episodes':len(rows),'expected_episodes':len(expected),'scenes':len(completed_scenes),
             'turn_deg':summaries[0].get('turn_deg',30),'max_steps':summaries[0]['max_steps'],
             'SR':mean(rows,'success'),'SPL':mean(rows,'spl'),'SoftSPL':mean(rows,'softspl'),
             'SR_scene_bootstrap_95':scene_ci(rows,'success'),'SPL_scene_bootstrap_95':scene_ci(rows,'spl'),
             'inference_ms_mean_during_concurrent_runs':mean(rows,'inference_ms_mean'),
             'collision_rate_mean_per_episode_forward_attempts':mean(rows,'collision_rate'),
             'timeout_rate':mean(rows,'timeout'),'stop_rate':mean(rows,'stopped'),
             'reached_success_region_rate':mean(rows,'reached_success_region'),'outcomes':dict(collections.Counter(r['outcome_tag'] for r in rows)),
             'not_policy_causal_diagnosis':True,'rows_sha256':hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest()}
    for field in ['distance_bin','detour_bin','category']:
        categories=sorted({r[field] for r in rows if r.get(field) is not None})
        summary['by_'+field]={c:{'episodes':len(a:=[r for r in rows if r.get(field)==c]),'SR':mean(a,'success'),'SPL':mean(a,'spl'),
                                'reached_success_region_rate':mean(a,'reached_success_region')} for c in categories}
    if rows[0]['task']=='instance_image_nav':
        summary['target_visible_ever_rate']=mean(rows,'target_visible_ever')
        summary['wrong_instance_stop_rate_selected_instances_only']=mean(rows,'wrong_instance_stop')
        summary['by_distractor_count']={name:{'episodes':len(a:=[r for r in rows if test(r['same_category_distractors'])]),'SR':mean(a,'success')}
              for name,test in [('0',lambda n:n==0),('1-3',lambda n:1<=n<=3),('4+',lambda n:n>=4)]}
    else:summary['strict_pose_success_rate']=mean(rows,'strict_pose_success')
    (folder/'aggregate.json').write_text(json.dumps(summary,indent=2));return summary

if __name__=='__main__':
    output=[]
    for folder in sorted((BASE/'results').iterdir()):
        if folder.is_dir() and not any(s in folder.name for s in ['smoke','oracle']):
            result=aggregate(folder)
            if result:output.append(result)
    (BASE/'results/baseline_comparison.json').write_text(json.dumps(output,indent=2))
    print(json.dumps([{k:r[k] for k in ['run','complete_frozen_panel','episodes','scenes','turn_deg','SR','SPL','outcomes']} for r in output],indent=2))
