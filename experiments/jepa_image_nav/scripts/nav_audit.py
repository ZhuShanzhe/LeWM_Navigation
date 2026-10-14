"""Audit saved task metadata, scene isolation, goal images and instance masks."""
import gzip,json,math,sys
from pathlib import Path
from collections import Counter
import numpy as np
from PIL import Image
def audit(root):
    root=Path(root);seen=set();scene_splits={};goals=set();counts=Counter();issues=[];strata=Counter()
    for path in sorted((root/'episodes').glob('*/*.jsonl.gz')):
        for line in gzip.open(path,'rt'):
            ep=json.loads(line);eid=ep['episode_id'];scene=ep['scene_id'];split=ep['split'];task=ep['task']
            if eid in seen:issues.append('duplicate episode '+eid)
            seen.add(eid)
            if scene in scene_splits and scene_splits[scene]!=split:issues.append('scene leakage '+scene)
            scene_splits[scene]=split;counts[f'{split}/{task}']+=1
            strata[f'{split}/{task}/distance/{ep["distance_bin"]}']+=1
            strata[f'{split}/{task}/detour/{ep["detour_bin"]}']+=1
            gap=abs((ep['start_yaw']-ep['goal_camera']['yaw']+math.pi)%(2*math.pi)-math.pi)*180/math.pi
            strata[f'{split}/{task}/start_goal_yaw/'+('0-45' if gap<45 else '45-90' if gap<90 else '90-180')]+=1
            if task=='instance_image_nav':strata[f'{split}/{task}/same_category_distractors/'+str(len(ep['same_category_distractors']))]+=1
            if not (1.5<=ep['geodesic_m']<=30):issues.append('invalid distance '+eid)
            for key in ('start_position','start_yaw'):
                if not np.all(np.isfinite(ep[key])):issues.append('nonfinite '+eid)
            goal=root/ep['goal_image']
            if goal not in goals:
                try:
                    im=np.asarray(Image.open(goal).convert('RGB'))
                    if im.shape!=(256,256,3) or im.std()<5:issues.append('invalid photo '+str(goal))
                    if task=='instance_image_nav':
                        mask=np.array(Image.open(goal.with_suffix('.mask.png')))>0
                        if mask.mean()<.025:issues.append('target too small '+str(goal))
                except Exception as e:issues.append('unreadable image '+str(goal)+' '+str(e))
                goals.add(goal)
            if task=='instance_image_nav' and (not ep['instance_id'].startswith(scene+':') or not ep['valid_approach_views']):issues.append('invalid instance '+eid)
    stats={'dataset':root.name,'episodes':len(seen),'scenes':len(scene_splits),'goal_images':len(goals),'split_task_counts':dict(counts),
           'stratification_counts':dict(strata),'issues':issues[:100],'issue_count':len(issues),'metadata_image_audit_passed':len(issues)==0,
           'geometric_validation':'Each episode was generated from an actual finite Habitat path; this audit does not replay all paths.',
           'not_model_performance':True}
    (root/'audit.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats,indent=2))
    return stats
if __name__=='__main__':
    if not audit(sys.argv[1])['metadata_image_audit_passed']:raise SystemExit(1)
