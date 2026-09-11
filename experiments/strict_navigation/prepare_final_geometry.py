"""Freeze four fresh geometry goals before final method evaluation; no model inference."""
from pathlib import Path
import json,hashlib,time
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
D=R/'final_geometry_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
prior=json.loads((R/'maps/catalog.json').read_text())
old_geometries={(m['axis'],m['door_center']) for m in prior['maps']}
assert all((a,c) not in old_geometries for a in [1,0] for c in [73,169])
D.mkdir(exist_ok=False)
source=(R/'prepare_maps.py').read_text()
replacements={"D=R/'maps'":"D=R/'final_geometry_v1'",
 "[('train',[49,81,113,145]),('validation',[65,129]),('test',[97,161,177])]":"[('final_geometry',[73,169])]",
 "9102030":"9106001"}
for a,b in replacements.items():
 assert a in source,a
 source=source.replace(a,b)
compile(source,'frozen_final_geometry_generator','exec')
(D/'generator_source.py').write_text(source)
exec(compile(source,'frozen_final_geometry_generator','exec'),{'__name__':'__main__'})
catalog=json.loads((D/'catalog.json').read_text())
assert len(catalog['maps'])==4 and all(x['oracle_reference_successes']==100 for x in catalog['audit'])
old_pairs=set()
for m in prior['maps']:
 for c in json.loads((R/'maps'/f"{m['id']}.json").read_text())['cases']:
  old_pairs.add(tuple(c['start_xy']+c['goals']['75']['xy']))
for m in catalog['maps']:
 cs=json.loads((D/f"{m['id']}.json").read_text())['cases']
 assert all(tuple(c['start_xy']+c['goals']['75']['xy']) not in old_pairs for c in cs)
manifest={'created_unix':time.time(),'purpose':'Fresh final geometric generalization confirmation after prior18 maps entered development; no model outcomes generated here.',
 'prior_geometry_catalog_sha256':sha(R/'maps/catalog.json'),'generator_sha256':sha(D/'generator_source.py'),
 'files_sha256':{p.name:sha(p) for p in D.glob('*.json')},
 'selection_seed':9106001,'geometries':[(a,c) for a in [1,0] for c in [73,169]],
 'all_prior_geometry_disjoint':True,'exact_coordinate_pairs_disjoint_from_prior_maps':True,
 'geometry_rationale':'For8-layout training,73 is inside observed door range49..145;169 is beyond it. Both are unseen geometry values, in both orientations. For strict original single-map49 models, both are unseen shifts, not interpolation.',
 'cases_per_map':100,'same_side':50,'cross_wall':50,'euclidean_range':[75,125],
 'same_coordinate_pairs_reused_across_four_maps':True,
 'limits':'Two rooms/one door only, not new topology. CPU privileged solvability only; no model scores, no final method selection here. Any later outcome-informed tuning invalidates final status.'}
(D/'freeze_manifest.json').write_text(json.dumps(manifest,indent=2))
print('FINAL_GEOMETRY_FROZEN',json.dumps({'maps':4,'cases_each':100,'audits':catalog['audit']}),flush=True)
