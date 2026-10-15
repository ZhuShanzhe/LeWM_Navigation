"""Freeze the accepted derived version; active file indexes are split-aware."""
import json,csv,hashlib,gzip
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014');REPORT=ROOT/'reports/acceptance_20261015'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    full=json.loads((REPORT/'full_acceptance.json').read_text());aux=json.loads((REPORT/'source_and_trajectory_audit.json').read_text())
    assert full['passed'] and aux['passed']
    out=ROOT/'datasets/hssd_image_nav_v1';protocol=json.loads((out/'protocol.json').read_text())
    protocol.update({'derived_revision':'20261015-compat1','rendering_scene_dataset_config':str(ROOT/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json'),
                     'upstream_scene_dataset_config':str(ROOT/'raw/hssd-hab/hssd-hab.scene_dataset_config.json'),
                     'material_compatibility_manifest':str(ROOT/'manifests/material_compatibility.json'),
                     'goal_image_formats':['JPEG','PNG'],'quality_gate':'decoded RGB standard deviation >=5, unchanged',
                     'active_data_selection':'Use episode/instance/trajectory indexes, NOT a recursive glob of raw image folders.',
                     'acceptance_report':str(REPORT/'full_acceptance.json')})
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    result={'date':'2026-10-15','derived_revision':'20261015-compat1','all_acceptance_checks_passed':True,
            'full_acceptance_sha256':sha(REPORT/'full_acceptance.json'),'source_and_trajectory_audit_sha256':sha(REPORT/'source_and_trajectory_audit.json'),
            'original_source_checksum_manifest_sha256':sha(ROOT/'manifests/hssd_local_source_sha256.csv'),'datasets':{}}
    for dataset in ['gibson_place_nav_v1','hssd_image_nav_v1']:
        d=ROOT/'datasets'/dataset;splits=json.loads((d/'split_manifest.json').read_text())['scenes'];active={}
        for pack in (d/'episodes').glob('*/*.jsonl.gz'):
            for line in gzip.open(pack,'rt'):
                ep=json.loads(line);active[ep['goal_image']]=ep['split']
        for instances in (d/'instances').glob('*.json'):
            split=splits[instances.stem]
            for target in json.loads(instances.read_text()):
                for view in target['views']:
                    active[view['image']]=split;active[str(Path(view['image']).with_suffix('.mask.png'))]=split
        for traj in (d/'trajectories').glob('*/*/trajectory.json'):
            record=json.loads(traj.read_text())
            if not record['success']:continue
            active[str(traj.relative_to(d))]='train'
            for frame in record['frames']:active[str((traj.parent/frame['image']).relative_to(d))]='train'
        index=ROOT/'manifests'/f'{dataset}_active_files.csv'
        with index.open('w',newline='') as f:
            writer=csv.writer(f);writer.writerow(['path','split','bytes','sha256'])
            for path,split in sorted(active.items()):
                file=d/path;writer.writerow([path,split,file.stat().st_size,sha(file)])
        result['datasets'][dataset]={'active_files':len(active),'active_file_index':str(index),'active_file_index_sha256':sha(index),
                                    'split_manifest_sha256':sha(d/'split_manifest.json'),'protocol_sha256':sha(d/'protocol.json')}
        print('ACTIVE_INDEX_FROZEN',dataset,len(active),flush=True)
    (REPORT/'release_freeze.json').write_text(json.dumps(result,indent=2));print('RELEASE_FROZEN',flush=True)
if __name__=='__main__':main()
