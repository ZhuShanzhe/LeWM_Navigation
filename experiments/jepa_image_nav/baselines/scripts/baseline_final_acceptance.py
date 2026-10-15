"""Validate complete evidence and report post-hoc constraint slices separately."""
import collections, datetime, gzip, hashlib, json, math, subprocess
from pathlib import Path
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015');DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')
EXPECTED={'fgprompt_gibson_val_place':280,'fgprompt_hssd_test_place':300,'fgprompt_hssd_test_instance':300,
          'debit_native10_gibson_val_place':280,'debit_native10_hssd_test_place':300,'debit_native10_hssd_test_instance':300,
          'debit_gibson_val_place':280,'lg_frontier_hssd_test_instance':300}

def main():
    repo='/root/autodl-tmp/lewm_research/github/LeWM_Navigation'
    frozen=json.loads(subprocess.check_output(['git','-C',repo,'show','HEAD:experiments/jepa_image_nav/baselines/results/panel_index.json'],text=True))
    for record in frozen:
        panel=BASE/'plans'/(record['panel']+'.jsonl')
        assert hashlib.sha256(panel.read_bytes()).hexdigest()==record['sha256'],'Frozen panel changed'
    comparison={r['run']:r for r in json.loads((BASE/'results/baseline_comparison.json').read_text())};checks=[];allrows={}
    for name,n in EXPECTED.items():
        a=comparison[name];assert a['complete_frozen_panel'] and a['episodes']==n
        folder=BASE/'results'/name;rows=[];traces=0
        for f in sorted(folder.glob('*/summary.json')):
            s=json.loads(f.read_text());ep=[json.loads(l) for l in (f.parent/'episodes.jsonl').read_text().splitlines()]
            assert s['status']=='complete' and len(ep)==s['episodes'] and sum(r['success'] for r in ep)==s['successes']
            assert s['panel_sha256']==a['panel_sha256'] and s['max_steps']==500 and s.get('turn_deg',30)==a['turn_deg']
            if s['model'] in ['fgprompt','debit']:
                official=json.loads((BASE/'results'/(s['model']+'_inference.json')).read_text())
                assert s['policy']['strict_state_dict'] and s['policy']['checkpoint_sha256']==official['checkpoint_sha256']
            counts=collections.Counter();last_actions={};last_positions={};trace_lengths=collections.defaultdict(float)
            with gzip.open(f.parent/'steps.jsonl.gz','rt') as g:
                for line in g:
                    st=json.loads(line);key=st['episode_id'];assert st['action'] in range(4)
                    assert st['t']==counts[key] and last_actions.get(key)!=0
                    if key in last_positions:trace_lengths[key]+=math.dist(last_positions[key],st['position'])
                    last_positions[key]=st['position'];last_actions[key]=st['action'];counts[key]+=1;traces+=1
            assert set(counts)=={r['episode_id'] for r in ep}
            for r in ep:
                assert 1<=r['steps']<=500 and counts[r['episode_id']]==r['steps']
                assert bool(r['stopped'])==(last_actions[r['episode_id']]==0)
                if r['stopped']:assert math.isclose(trace_lengths[r['episode_id']],r['executed_length_m'],rel_tol=1e-6,abs_tol=1e-5)
                else:assert -.00001<=r['executed_length_m']-trace_lengths[r['episode_id']]<=.5
                inst=r['task']=='instance_image_nav'
                success=bool(r['stopped'] and r['final_distance_m']<(.5 if inst else 1.) and (not inst or r['target_visible_at_stop']))
                assert success==bool(r['success'])
                spl=int(success)*r['initial_distance_m']/max(r['initial_distance_m'],r['executed_length_m'],1e-8)
                assert math.isclose(spl,r['spl'],rel_tol=1e-7,abs_tol=1e-8)
                assert all(math.isfinite(r[k]) for k in ['spl','softspl','final_distance_m','executed_length_m'])
            rows.extend(ep)
        panel=[json.loads(l) for l in (BASE/'plans'/a['panel']).read_text().splitlines()]
        assert len(rows)==n and len({r['episode_id'] for r in rows})==n
        assert {r['episode_id'] for r in rows}=={e['episode_id'] for e in panel}
        assert math.isclose(sum(r['success'] for r in rows)/n,a['SR'])
        if rows[0]['task']=='instance_image_nav':
            assert math.isclose(sum(bool(r.get('wrong_instance_stop')) and not r['success'] for r in rows)/n,a['wrong_instance_stop_rate_selected_instances_only'])
        checks.append({'run':name,'episodes':n,'scenes':a['scenes'],'trace_steps_verified':traces,'SR':a['SR'],'SPL':a['SPL'],'passed':True});allrows[name]=rows
    visibility=json.loads((BASE/'results/horizontal_visibility_audit.json').read_text());assert visibility['target_instances']==319 and visibility['frozen_panel_episodes']==300
    lookup={r['instance_id']:r for r in visibility['per_instance']};ds=DATA/'datasets/hssd_image_nav_v1'
    splits=json.loads((ds/'split_manifest.json').read_text())['scenes'];trainassets=set();targets={}
    for f in (ds/'instances').glob('*.json'):
        for t in json.loads(f.read_text()):
            targets[t['instance_id']]=t
            if splits[f.stem]=='train':trainassets.add(t['asset_id'])
    panel={e['episode_id']:e for e in [json.loads(l) for l in (BASE/'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl').read_text().splitlines()]}
    slices=[]
    for name,rows in allrows.items():
        if not name.endswith('hssd_test_instance'):continue
        groups=collections.defaultdict(list)
        for r in rows:
            e=panel[r['episode_id']];t=targets[e['instance_id']]
            groups['yaw_stored_approach_verified' if lookup[e['instance_id']]['yaw_only_visible_at_a_stored_approach_position'] else 'yaw_stored_approach_unverified'].append(r)
            groups['asset_seen_in_selected_train_targets' if t['asset_id'] in trainassets else 'asset_not_seen_in_selected_train_targets'].append(r)
        slices.append({'run':name,'groups':{k:{'episodes':len(v),'successes':sum(r['success'] for r in v),'SR':sum(r['success'] for r in v)/len(v),'SPL':sum(r['spl'] for r in v)/len(v)} for k,v in groups.items()}})
    (BASE/'results/constraint_slices.json').write_text(json.dumps({'post_hoc_diagnostic':True,'primary_panel_unchanged':True,'scope':'Stored horizontal approaches only; unverified does not prove unreachable. Selected target asset overlap only, not all scene assets. No slice replaces primary score.','runs':slices},indent=2))
    latency=json.loads((BASE/'results/isolated_latency.json').read_text());assert latency['status']=='complete' and len(latency['runs'])==2
    camera=json.loads((BASE/'results/camera_extrinsics_audit.json').read_text());assert camera['status']=='passed' and camera['scenes']==168
    assert len(camera['live_sensor_transform_checks'])==6
    result={'status':'passed','captured_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
            'checks':checks,'RGB_primary_model_episode_executions':1760,'interface_diagnostic_executions':280,'RGBD_auxiliary_executions':300,
            'frozen_unique_task_episodes':880,'visibility_instances_verified':319,'isolated_latency_complete':True,
            'goal_camera_extrinsics_sidecar_scenes':camera['scenes'],'goal_camera_extrinsics_sidecar_images':camera['target_images'],
            'limitations':['Specialized IIN full system not reproduced; ImageNav transfer and custom RGBD matcher are not dedicated IIN state of the art.','Exclusive latency replay is not equal-time-budget navigation.','Raw dataset acceptance is separate from policy competence.']}
    (BASE/'results/final_acceptance.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
