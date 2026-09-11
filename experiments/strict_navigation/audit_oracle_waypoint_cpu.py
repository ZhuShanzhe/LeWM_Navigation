"""Finite CPU checks for oracle route feasibility and non-mutating goal-image substitution."""
from pathlib import Path
import json,time
import numpy as np,torch
from topology_env import TopologyEnv
from oracle_waypoint_router import OracleWaypointRouter
torch.set_num_threads(1)
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
records=[]
for mapid in ['four_room_chain','four_room_cycle','six_room_branch']:
    data=json.loads((R/'topology_dev_v2'/f'{mapid}.json').read_text());cases=data['cases'];spec=data['topology_spec']
    router=OracleWaypointRouter(spec,cases);env=TopologyEnv(spec);env.reset(seed=9104401);steps=[];segments=0
    for c,route in zip(cases,router.routes):
        env._set_state(np.array(c['start_xy'],np.float32));env._set_goal_state(np.array(c['goals']['75']['xy'],np.float32))
        n=0;success=False
        for point in route:
            segments+=1
            while np.linalg.norm(env.agent_position.numpy()-point)>.05:
                _,_,success,_,_=env.step(np.clip((point-env.agent_position.numpy())/5,-1,1).astype(np.float32));n+=1
                assert env.valid(env.agent_position.numpy()) and n<=150
                if success:break
            if success:break
        assert success;steps.append(n)
    chosen=[cases[0],cases[99]]
    finals=[]
    for c in chosen:
        env._set_state(np.array(c['goals']['75']['xy'],np.float32));finals.append(env.render().copy())
    info={'goal':np.stack(finals)[:,None],'proprio':np.array([c['start_xy'] for c in chosen],np.float32)[:,None],'terminated':np.zeros(2,bool)}
    before={k:v.copy() for k,v in info.items()}
    final_router=OracleWaypointRouter(spec,chosen,mode='final');out=final_router.transform(info)
    assert np.array_equal(out['goal'],info['goal'])
    waypoint_router=OracleWaypointRouter(spec,chosen);out=waypoint_router.transform(info)
    assert all(np.array_equal(info[k],before[k]) for k in info)
    # Dead environments retain their original goal pixels and do not progress.
    info['terminated'][:]=True;ix=waypoint_router.indices.copy();out=waypoint_router.transform(info)
    assert np.array_equal(out['goal'],info['goal']) and np.array_equal(ix,waypoint_router.indices)
    records.append({'map':mapid,'cases':100,'reference_successes':100,'max_reference_steps':max(steps),'route_segments':segments,'final_passthrough_pixels_exact':True,'input_unmodified':True,'dead_state_unchanged':True})
    env.close();print('ORACLE_CPU_MAP_PASSED',mapid,flush=True)
result={'passed':True,'updated_unix':time.time(),'records':records,'limits':'CPU scaffold/interface proof only; no learned policy result. True geometry and pose are privileged.'}
(R/'reports/oracle_waypoint_cpu_audit.json').write_text(json.dumps(result,indent=2));print('ORACLE_CPU_AUDIT_PASSED',flush=True)
