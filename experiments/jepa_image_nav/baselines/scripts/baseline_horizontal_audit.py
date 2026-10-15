"""Check target visibility at stored approach positions with yaw-only cameras.

This does not change the benchmark or discard failures. It identifies a
controller/observation constraint independently of policy predictions.
"""
import json, os, sys
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015');DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')
sys.path.insert(0,str(DATA/'tools'))
from nav_build import opensim,setpose

if __name__=='__main__':
    panel=[json.loads(l) for l in (BASE/'plans/hssd_image_nav_v1_test_instance_20perscene.jsonl').read_text().splitlines()]
    scenes=sorted({e['scene_id'] for e in panel});allrows=[]
    for scene in scenes:
        out=BASE/'results/horizontal_visibility'/f'{scene}.json'
        if out.exists():allrows.extend(json.loads(out.read_text()));continue
        targets=json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/f'{scene}.json').read_text())
        sim=opensim(scene,DATA/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json');rows=[]
        try:
            objs=sim.get_rigid_object_manager().get_objects_by_handle_substring('')
            for i,(h,o) in enumerate(sorted(objs.items()),1):o.semantic_id=i
            for target in targets:
                best=0.;n=0;valid=None
                for view in target['valid_approach_views']:
                    obs=setpose(sim,view['position'],view['yaw'],0.);n+=1
                    fraction=float((obs['semantic']==target['semantic_id']).mean());best=max(best,fraction)
                    if fraction>=.025:valid=view;break
                rows.append({'scene':scene,'instance_id':target['instance_id'],'yaw_only_visible_at_a_stored_approach_position':valid is not None,
                             'views_checked_before_first_success':n,'best_target_fraction':best,
                             'approach_pitch_range_deg':[min(v['pitch'] for v in target['valid_approach_views'])*180/np.pi,max(v['pitch'] for v in target['valid_approach_views'])*180/np.pi]})
        finally:sim.close()
        out.parent.mkdir(exist_ok=True);out.write_text(json.dumps(rows,indent=2));allrows.extend(rows)
    lookup={r['instance_id']:r for r in allrows};bad=[r for r in allrows if not r['yaw_only_visible_at_a_stored_approach_position']]
    result={'target_instances':len(allrows),'without_a_verified_yaw_only_approach_position':len(bad),
            'frozen_panel_episodes':len(panel),'panel_episodes_without_verified_yaw_only_approach':sum(not lookup[e['instance_id']]['yaw_only_visible_at_a_stored_approach_position'] for e in panel),
            'failures':bad,'per_instance':allrows,
            'scope':'At exact stored approach positions facing the object with pitch=0; not proof no other navigable position works; no changes to primary episode panel or scoring.'}
    (BASE/'results/horizontal_visibility_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['per_instance','failures']}),flush=True)
