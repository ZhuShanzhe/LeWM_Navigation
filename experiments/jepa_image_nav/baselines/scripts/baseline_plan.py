"""Freeze scene-balanced evaluation panels before any model results are seen."""
import gzip, hashlib, json
from collections import defaultdict, Counter
from pathlib import Path

ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014')
OUT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015/plans')

def stable(s):return hashlib.sha256(s.encode()).hexdigest()

def panel(dataset,split,task,per_scene):
    source=ROOT/'datasets'/dataset
    selected=[];scene_counts={}
    for f in sorted((source/'episodes'/split).glob('*.jsonl.gz')):
        if dataset.startswith('hssd') and not f.name.endswith('.'+task+'.jsonl.gz'):continue
        with gzip.open(f,'rt') as r:eps=[json.loads(line) for line in r if line.strip()]
        groups=defaultdict(list)
        for e in eps:groups[(e['distance_bin'],e['detour_bin'])].append(e)
        for key,values in groups.items():values.sort(key=lambda e:stable('eval-v1-'+e['episode_id']))
        chosen=[];seen=set();keys=sorted(groups)
        while len(chosen)<per_scene and any(groups.values()):
            for key in keys:
                if len(chosen)>=per_scene:break
                group=groups[key]
                if not group:continue
                # Prefer different goal photographs, then allow repeats only if needed.
                k=next((i for i,e in enumerate(group) if e['goal_image'] not in seen),0)
                e=group.pop(k);chosen.append(e);seen.add(e['goal_image'])
        selected.extend(chosen)
        if chosen:scene_counts[chosen[0]['scene_id']]=len(chosen)
    name=f'{dataset}_{split}_{task}_{per_scene}perscene'
    target=OUT/(name+'.jsonl')
    text=''.join(json.dumps(dict(dataset=dataset,**e),sort_keys=True)+'\n' for e in selected)
    if target.exists():assert target.read_text()==text,'Refuse to change frozen panel'
    else:target.write_text(text)
    meta={'panel':name,'sha256':hashlib.sha256(text.encode()).hexdigest(),'episodes':len(selected),
          'scenes':scene_counts,'sampling':'deterministic round-robin distance x detour bins, scene balanced; not natural-frequency distribution',
          'counts':dict(Counter(e['distance_bin'] for e in selected)),
          'training_overlap_caveat':('Gibson local test reserved from original train: possibly seen by released navigation weights' if dataset.startswith('gibson') and split=='test' else None)}
    (OUT/(name+'.json')).write_text(json.dumps(meta,indent=2))
    return meta

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    out=[]
    for ds,split,task,n in [('gibson_place_nav_v1','val','place',20),('hssd_image_nav_v1','test','place',20),
                            ('hssd_image_nav_v1','test','instance',20),('hssd_image_nav_v1','val','instance',2),
                            ('hssd_image_nav_v1','val','place',2)]:
        out.append(panel(ds,split,task,n))
    (OUT/'panel_index.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out,indent=2))
