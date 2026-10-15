"""Create an explicit derived overlay, never overwrite pinned source assets."""
import json,struct,hashlib,copy,shutil
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014')
SRC=ROOT/'raw/hssd-hab';DST=ROOT/'raw/hssd-hab-compat'
ASSET='objects/x/xxxx21d2f3c8xedd4x4619x90b0x62bad36c41cb.glb'
def sha(b):return hashlib.sha256(b).hexdigest()
def main():
    original=(SRC/ASSET).read_bytes();n,kind=struct.unpack_from('<II',original,12)
    assert kind==0x4e4f534a
    doc=json.loads(original[20:20+n]);before=copy.deepcopy(doc)
    mat=doc['materials'][0];pbr=mat['pbrMetallicRoughness'];transform=pbr['baseColorTexture']['extensions']['KHR_texture_transform']
    assert mat['normalTexture']['extensions']['KHR_texture_transform']==transform
    assert 'extensions' not in pbr['metallicRoughnessTexture']
    pbr['metallicRoughnessTexture']['extensions']={'KHR_texture_transform':copy.deepcopy(transform)}
    encoded=json.dumps(doc,separators=(',',':')).encode();encoded+=b' '*((-len(encoded))%4)
    tail=original[20+n:];repaired=struct.pack('<III',0x46546c67,2,20+len(encoded)+len(tail))+struct.pack('<II',len(encoded),kind)+encoded+tail
    DST.mkdir(exist_ok=True)
    # All originals stay untouched. Only the one GLB and the root config are copied.
    for p in SRC.iterdir():
        if p.name in ['objects','.cache']:continue
        dest=DST/p.name
        if p.name=='hssd-hab.scene_dataset_config.json':
            if not dest.exists():shutil.copy2(p,dest)
        elif not dest.exists():dest.symlink_to(p,target_is_directory=p.is_dir())
    (DST/'objects').mkdir(exist_ok=True)
    for p in (SRC/'objects').iterdir():
        dest=DST/'objects'/p.name
        if p.name!='x':
            if not dest.exists():dest.symlink_to(p,target_is_directory=p.is_dir())
        else:
            dest.mkdir(exist_ok=True)
            for f in p.iterdir():
                target=dest/f.name
                if str(f.relative_to(SRC))==ASSET:continue
                if not target.exists():target.symlink_to(f,target_is_directory=f.is_dir())
    target=DST/ASSET
    assert not target.is_symlink()
    if target.exists():assert target.read_bytes()==repaired,'Existing compatibility revision differs'
    else:
        tmp=target.with_suffix('.glb.tmp');tmp.write_bytes(repaired);tmp.replace(target)
    assert (SRC/ASSET).read_bytes()==original
    assert repaired[20+len(encoded):]==tail
    check=copy.deepcopy(doc);del check['materials'][0]['pbrMetallicRoughness']['metallicRoughnessTexture']['extensions']
    assert check==before
    report={'source_revision':'4369cb9876214c7fbebcf552eb532380e4d287e4','asset':ASSET,'original_sha256':sha(original),'derived_sha256':sha(repaired),
            'change':'Set metallicRoughnessTexture KHR_texture_transform to the existing shared base-color/normal transform.',
            'appearance_limit':'Metallic/roughness sampling coordinates change. This is an explicit compatibility revision, NOT pixel-equivalent to the original PBR material.',
            'geometry_buffers_images_collision_unchanged':True,'original_source_unchanged':True,'overlay':str(DST)}
    handle=Path(ASSET).stem
    report['referencing_scenes']=[p.name.split('.scene_instance')[0] for p in sorted((SRC/'scenes').glob('*.json')) if any(o['template_name'].split('/')[-1]==handle for o in json.loads(p.read_text())['object_instances'])]
    (ROOT/'manifests/material_compatibility.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
