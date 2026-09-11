"""Optional fixed diagnostic cases; leaves official evaluation pipeline and solver intact."""
import os,json
from pathlib import Path
import numpy as np
import stable_worldmodel as swm
if os.environ.get('LEWM_NOOP')=='1':
 def noop(self,obs,**kwargs):return np.zeros(self.env.action_space.shape,dtype=self.env.action_space.dtype)
 swm.policy.RandomPolicy.get_action=noop
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909')
source=(R/'eval_record.py').read_text()
injection=""" case_path=os.environ.get('LEWM_CASE_FILE')
 if case_path:
  case_data=json.loads(Path(case_path).read_text());cases=case_data['cases'][:int(os.environ.get('LEWM_CASE_LIMIT','50'))]
  kw['episodes_idx']=[v['episode'] for v in cases];kw['start_steps']=[v['start_step'] for v in cases]
  case_data.update({'cases':cases,'goal_offset':kw.get('goal_offset'),'budget':kw.get('eval_budget'),'noop':os.environ.get('LEWM_NOOP')=='1'})
  (out/'case_metadata.json').write_text(json.dumps(case_data,indent=2))
"""
assert " kw['video']=" in source
source=source.replace(" kw['video']=",injection+" kw['video']=",1)
exec(compile(source,str(R/'eval_focused.py'),'exec'),{'__name__':'__main__','__file__':str(R/'eval_focused.py')})
