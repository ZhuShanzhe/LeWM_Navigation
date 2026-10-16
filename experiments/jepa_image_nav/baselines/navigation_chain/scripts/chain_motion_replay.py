"""Replay unusual translations through the actual installed Habitat controls."""
import argparse
import fcntl
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import quaternion
from instance_eval_goat import ROOT, PROFILES, opensim, write

parser=argparse.ArgumentParser()
parser.add_argument('--requests',required=True)
parser.add_argument('--output',required=True)
args=parser.parse_args()
requests=json.loads(Path(args.requests).read_text())
grouped=defaultdict(list)
for request in requests:
    grouped[(request['scene'],request['profile'])].append(request)
records=[]
with (ROOT/'gpu_benchmark.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    for (scene,profile),items in grouped.items():
        sim=opensim(scene,PROFILES[profile])
        try:
            agent=sim.get_agent(0)
            for request in items:
                state=agent.get_state()
                state.position=np.asarray(request['before'],dtype=np.float32)
                q=request['rotation_xyzw']
                state.rotation=quaternion.quaternion(q[3],*q[:3])
                agent.set_state(state)
                agent.act('move_forward')
                after=np.asarray(agent.get_state().position)
                error=float(np.linalg.norm(after-np.asarray(request['after'])))
                records.append({'episode_id':request['episode_id'],'t':request['t'],
                    'recorded_displacement_m':float(np.linalg.norm(np.asarray(request['after'])-request['before'])),
                    'replay_error_m':error,'passed':error<2e-5})
        finally:
            sim.close()
result={'passed':all(r['passed'] for r in records),'cases':len(records),
    'maximum_replay_error_m':max((r['replay_error_m'] for r in records),default=0.),
    'reason':'Forward displacement may exceed 0.25m in 3D due to navmesh projection, slopes and sliding; verify actual native control instead of changing scores',
    'records':records}
write(Path(args.output),result)
print(json.dumps({k:v for k,v in result.items() if k!='records'}),flush=True)
raise SystemExit(0 if result['passed'] else 1)
