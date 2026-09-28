"""Frozen default control across all 15 procedural terrain families, no tuning."""
import json,hashlib
from pathlib import Path
import deployment_eval_v2 as ev
from deployment_suite import summarize
out=ev.ROOT/'runs/deployment_v2_extended135';out.mkdir(exist_ok=False)
manifest=ev.ROOT/'datasets/procedural_v1/manifests/validation.jsonl'
cases=[json.loads(s) for s in manifest.read_text().splitlines()]
cases=[c for c in cases if c['condition']=='clean' and c['pair_id'].endswith('_T0')]
assert len(cases)==135
for c in cases:
    c.update(action_mode='deployment_default',group=f"{c['terrain']}_L{c['difficulty']}")
    c['scene_sha256']=hashlib.sha256(Path(c['scene_file']).read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(cases,indent=2))
for source in [Path(__file__),Path(ev.__file__)]:
    (out/source.name).write_bytes(source.read_bytes())
policy=ev.Policy();rows=[]
for i,c in enumerate(cases):
    r,t=ev.run(c,policy,True);rows.append(r)
    with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    (out/(c['episode_id']+'.trace.json')).write_text(json.dumps(t))
    summarize(out,rows)
    print(json.dumps({'i':i,'total':len(cases),'group':c['group'],'reason':r['reason'],'wall_s':r['wall_seconds']}),flush=True)
(out/'complete').write_text('complete\n')
