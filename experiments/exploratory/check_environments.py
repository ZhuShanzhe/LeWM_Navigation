import json,traceback
from pathlib import Path
import numpy as np,gymnasium as gym,stable_worldmodel
tests=[('pusht','swm/PushT-v1',{}),('cube','swm/OGBCube-v0',{'env_type':'single','ob_type':'states','multiview':False,'width':224,'height':224,'visualize_info':False,'terminate_at_goal':True}),('reacher','swm/ReacherDMControl-v0',{'task':'qpos_match'})]
out=[]
for label,name,kw in tests:
 try:
  env=gym.make(name,**kw);obs,info=env.reset(seed=42);result=env.step(env.action_space.sample());img=env.render()
  row={'env':label,'ok':True,'render_shape':list(np.asarray(img).shape),'info_keys':list(info),'action_shape':list(env.action_space.shape)}
  env.close()
 except Exception as e:row={'env':label,'ok':False,'error':repr(e),'traceback':traceback.format_exc()}
 out.append(row);print(row,flush=True)
Path('/root/autodl-tmp/lewm_research/round12h_20260909/environment_checks.json').write_text(json.dumps(out,indent=2))
