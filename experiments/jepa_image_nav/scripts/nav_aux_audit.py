"""Local source freeze and complete expert trajectory image/causality audit."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014');OUT=ROOT/'reports/acceptance_20261015'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        while True:
            chunk=f.read(4*1024*1024)
            if not chunk:break
            h.update(chunk)
    return h.hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True);source=json.loads((ROOT/'manifests/hssd_source.json').read_text());missing=[];total=0
    with (ROOT/'manifests/hssd_local_source_sha256.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['path','bytes','sha256'])
        for entry in source['files']:
            p=ROOT/'raw/hssd-hab'/entry['path']
            if not p.is_file() or p.stat().st_size==0:missing.append(entry['path']);continue
            w.writerow([entry['path'],p.stat().st_size,sha(p)]);total+=p.stat().st_size
    print('SOURCE_FREEZE_COMPLETE',len(source['files']),total,flush=True)
    records=[];problems=[];frames=0;trajectories=0
    for dataset in ['gibson_place_nav_v1','hssd_image_nav_v1']:
        root=ROOT/'datasets'/dataset;expected=sum(len(json.loads(p.read_text())['expert_trajectories']) for p in (root/'scene_statistics').glob('*.json'))
        files=list((root/'trajectories').glob('*/*/trajectory.json'))
        if len(files)!=expected:problems.append(f'{dataset}: trajectory count {len(files)} != {expected}')
        accepted=[];failed=[]
        for p in files:
            d=json.loads(p.read_text());trajectories+=1
            if d['split']!='train':problems.append('nontrain trajectory '+str(p))
            seq=d['frames']
            if not seq:problems.append('empty trajectory '+str(p))
            for i,frame in enumerate(seq):
                a=np.asarray(Image.open(p.parent/frame['image']).convert('RGB'));frames+=1
                if a.shape!=(256,256,3):problems.append('invalid trajectory frame '+str(p))
                if frame['t']!=i:problems.append('nonsequential time '+str(p))
                if i+1<len(seq):
                    q=seq[i+1]
                    if 'executed_translation_world' not in frame:problems.append('missing executed action '+str(p));continue
                    if not np.allclose(np.array(frame['position'])+frame['executed_translation_world'],q['position'],atol=1e-5):problems.append('position transition mismatch '+str(p))
                    if not np.allclose(frame['next_rotation_xyzw'],q['rotation_xyzw'],atol=1e-5):problems.append('rotation transition mismatch '+str(p))
            if d['success']:accepted.append(str(p.relative_to(root)))
            else:failed.append(str(p.relative_to(root)))
        index={'dataset':dataset,'training_only':True,'accepted_successful_trajectories':accepted,'excluded_unsuccessful_trajectories':failed}
        (OUT/f'{dataset}_trajectory_index.json').write_text(json.dumps(index,indent=2))
        records.append({'dataset':dataset,'expected_trajectories':expected,'actual_trajectories':len(files),'accepted_successful':len(accepted),'excluded_unsuccessful':len(failed)})
        print('TRAJECTORY_AUDIT',records[-1],flush=True)
    report={'passed':not missing and not problems,'source_files_checked':len(source['files']),'source_bytes':total,'source_missing_or_empty':missing,
            'source_sha256_scope':'Local freeze of all pinned-source files; not an independent upstream checksum comparison.',
            'trajectories_checked':trajectories,'frames_checked':frames,'datasets':records,'issues':problems[:100],'issue_count':len(problems)}
    (OUT/'source_and_trajectory_audit.json').write_text(json.dumps(report,indent=2));print('AUX_AUDIT_PASSED',report['passed'],flush=True)
    if not report['passed']:raise SystemExit(1)
if __name__=='__main__':main()
