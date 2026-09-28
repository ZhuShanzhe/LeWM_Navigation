"""Paired rotation diagnostics: isolate smoothing from hip scaling, simulation only."""
import json
from pathlib import Path
import deployment_eval_v2 as ev
from deployment_suite import summarize
class AblationPolicy(ev.Policy):
    def step(self,obs,depth,k,reset_memory=False):
        previous=self.action.copy()
        # The base only applies processing for deployment_default. Ablation names
        # leave raw model output here, while preserving raw-action observation history.
        super().step(obs,depth,k,reset_memory)
        applied=self.action.copy()
        if self.action_mode=='smooth_only':applied=.8*applied+.2*previous
        elif self.action_mode=='hip_only':applied[:4]*=.5
        else:raise ValueError(self.action_mode)
        return ev.Q0+.25*applied
out=ev.ROOT/'runs/deployment_v2_rotation_ablation20';out.mkdir(exist_ok=False)
original=json.loads((ev.ROOT/'runs/deployment_v2_230/manifest.json').read_text())
cases=[]
for c in original:
    if c['action_mode']=='deployment_default' and c['group'] in ['rotate_left','rotate_right']:
        for mode in ['smooth_only','hip_only']:cases.append({**c,'action_mode':mode,'episode_id':c['episode_id']+'_'+mode})
assert len(cases)==20
(out/'manifest.json').write_text(json.dumps(cases,indent=2))
(out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
rows=[];policy=AblationPolicy()
for i,c in enumerate(cases):
    r,t=ev.run(c,policy,True);rows.append(r)
    with (out/'results.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    (out/(c['episode_id']+'.trace.json')).write_text(json.dumps(t))
    summarize(out,rows);print(json.dumps({'i':i,'group':c['group'],'mode':c['action_mode'],'yaw_rmse':r['tracking_yaw_rmse']}),flush=True)
(out/'complete').write_text('complete\n')
