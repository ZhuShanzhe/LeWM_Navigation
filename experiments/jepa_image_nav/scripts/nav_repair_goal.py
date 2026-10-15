"""Re-render the same camera losslessly; preserve all twenty task geometries."""
import gzip,json,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image
from nav_build import ROOT,opensim,setpose
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=ROOT/'datasets/hssd_image_nav_v1';scene='105515307_173104317'
    oldrel=f'images/{scene}/place_0025.jpg';newrel=f'images/{scene}/place_0025.png'
    episode_file=out/'episodes/test'/f'{scene}.place.jsonl.gz'
    episodes=[json.loads(line) for line in gzip.open(episode_file,'rt')]
    affected=[e for e in episodes if e['goal_image']==oldrel]
    if not affected:
        assert len([e for e in episodes if e['goal_image']==newrel])==20
        print('GOAL_ALREADY_REPAIRED');return
    assert len(affected)==20
    cam=affected[0]['goal_camera'];assert all(e['goal_camera']==cam for e in affected)
    sim=opensim(scene,ROOT/'raw/hssd-hab/hssd-hab.scene_dataset_config.json')
    try:rgb=setpose(sim,cam['position'],cam['yaw'],cam['pitch'])['rgb'][:,:,:3].copy()
    finally:sim.close()
    print('RERENDER_RAW_STD',float(rgb.std()),flush=True)
    assert rgb.std()>=5,'Re-render failed quality gate; no episode modification performed'
    backup=ROOT/'provenance/goal_repair_20261015';backup.mkdir(parents=True,exist_ok=True)
    for f in [episode_file,out/oldrel]:
        target=backup/f.name
        if not target.exists():shutil.copy2(f,target)
    original_sha=sha(episode_file)
    tmp=(out/newrel).with_suffix('.png.tmp');Image.fromarray(rgb).save(tmp,format='PNG');tmp.replace(out/newrel)
    assert np.array(Image.open(out/newrel)).std()>=5
    for ep in affected:ep['goal_image']=newrel
    tmp=episode_file.with_suffix('.gz.tmp')
    with gzip.open(tmp,'wt') as f:
        for e in episodes:f.write(json.dumps(e)+'\n')
    tmp.replace(episode_file)
    report={'scene':scene,'affected_episodes':20,'episode_ids':[e['episode_id'] for e in affected],
            'old_goal_image':oldrel,'new_goal_image':newrel,'old_jpeg_std':float(np.array(Image.open(out/oldrel).convert('RGB')).std()),
            'new_png_std':float(rgb.std()),'quality_threshold_unchanged':5,'camera_start_goal_split_ids_unchanged':True,
            'action':'Re-render identical camera to lossless PNG; only goal_image references change.',
            'original_episode_pack_sha256':original_sha,'new_episode_pack_sha256':sha(episode_file),'new_png_sha256':sha(out/newrel),'backup':str(backup)}
    (ROOT/'manifests/goal_repair.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
