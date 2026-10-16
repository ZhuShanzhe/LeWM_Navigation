"""Habitat-Sim process serving only the native UniGoal sensor contract.

Ground-truth target geometry and semantic IDs stay in this scoring process.
The inference process receives RGB, normalized depth, relative GPS/compass,
body pose and the supplied goal image; category is sent only if explicitly
requested by the category-assisted protocol.
"""
import argparse
import contextlib
import gzip
import json
import math
from pathlib import Path
import sys
import traceback
import numpy as np
from PIL import Image
from instance_eval_goat import ROOT, DATA, OLD, PROFILES, opensim, sensor_values, write
sys.path.insert(0,str(DATA/'tools'))
from nav_build import setpose,pathlength
from habitat_sim.utils.common import quat_to_coeffs

parser=argparse.ArgumentParser()
parser.add_argument('--panel',required=True)
parser.add_argument('--output',required=True)
parser.add_argument('--buffer',required=True)
parser.add_argument('--category-assisted',action='store_true')
parser.add_argument('--max-steps',type=int,default=500)
args=parser.parse_args()
out=Path(args.output)
out.mkdir(parents=True,exist_ok=True)
buffer=Path(args.buffer)
panel=[json.loads(x) for x in Path(args.panel).read_text().splitlines()]
profile=dict(PROFILES['shared'],resolution=[640,480],hfov=79,camera_height=.88)
sim=None
scene=None
index=-1
records=[]
terminal=False
def emit(value):
    print(json.dumps(value),flush=True)
def transmit():
    from habitat_sim import SensorType,CameraSensorSpec
    state=sim.get_agent(0).get_state()
    gps,compass=sensor_values(state.position,state.rotation,origin,start_rotation)
    depth=obs['depth'][...,None].astype(np.float32)
    assert np.isfinite(depth).all() and depth.min()>=0
    depth=np.clip(depth,0,30)/30.
    np.savez(buffer,rgb=obs['rgb'][:,:,:3],depth=depth,gps=np.asarray(gps,np.float32),
        compass=np.asarray([compass],np.float32),position=state.position,
        rotation_xyzw=quat_to_coeffs(state.rotation),goal_rgb=goal)
    info={'time':step,'sensor_pose':[0.,0.,0.],
        'agent_height':.88+float(state.position[1]-origin[1]),'episode_no':index,
        'success':None,'spl':None,'distance_to_goal':None}
    if terminal:
        info.update(success=records[-1]['success'],spl=records[-1]['spl'],distance_to_goal=records[-1]['final_distance_m'])
    value={'buffer':str(buffer),'done':terminal,'info':info}
    if args.category_assisted:
        value['supplied_category']=e['category']
    emit(value)
try:
    for line in sys.stdin:
        request=json.loads(line)
        if request['op']=='close':
            break
        if request['op']=='reset':
            if terminal and index+1==len(panel):
                transmit()
                continue
            index+=1
            assert index<len(panel)
            e=panel[index]
            if scene!=e['scene_id']:
                if sim is not None:
                    sim.close()
                scene=e['scene_id']
                # Keep the frozen common navmesh/body. Native camera and actions
                # are kept, but this is explicitly NOT native HM3D geometry.
                import habitat_sim as hs
                import magnum as mn
                original_opensim=opensim
                with contextlib.redirect_stdout(sys.stderr):
                    sim=opensim(scene,profile)
                # Add a registered depth sensor to the existing native camera rig.
                spec=hs.CameraSensorSpec()
                spec.uuid='depth'
                spec.sensor_type=hs.SensorType.DEPTH
                spec.resolution=profile['resolution']
                spec.position=mn.Vector3(0,.88,0)
                spec.hfov=mn.Deg(79)
                sim.add_sensor(spec)
                sim.get_agent(0).agent_config.action_space['turn_left'].actuation.amount=15
                sim.get_agent(0).agent_config.action_space['turn_right'].actuation.amount=15
            targets=json.loads((DATA/'datasets/hssd_image_nav_v1/instances'/(scene+'.json')).read_text())
            target=next(t for t in targets if t['instance_id']==e['instance_id'])
            goal=np.asarray(Image.open(DATA/'datasets/hssd_image_nav_v1'/e['goal_image']).convert('RGB'))
            obs=setpose(sim,e['start_position'],e['start_yaw'])
            initial=sim.get_agent(0).get_state()
            origin=np.asarray(initial.position).copy()
            start_rotation=initial.rotation
            assert sim.pathfinder.is_navigable(origin,max_y_delta=.05)
            def distance(position):
                values=[pathlength(sim,position,v['position']) for v in e['valid_approach_views']]
                return min((d for d in values if d is not None and np.isfinite(d)),default=1e6)
            initial_distance=distance(origin)
            assert initial_distance<1e6
            length=0.
            step=0
            terminal=False
            trace=(out/(e['episode_id']+'.steps.jsonl')).open('w')
            transmit()
        else:
            assert request['op']=='step' and set(request)=={'op','action'}
            assert not terminal
            action=int(request['action'])
            assert action in {0,1,2,3}
            state=sim.get_agent(0).get_state()
            before=np.asarray(state.position).copy()
            before_rotation=quat_to_coeffs(state.rotation).tolist()
            if action:
                obs=sim.step({1:'move_forward',2:'turn_left',3:'turn_right'}[action])
            after=np.asarray(sim.get_agent(0).get_state().position).copy()
            length+=float(np.linalg.norm(after-before))
            fraction=float((obs['semantic']==target['semantic_id']).mean())
            step+=1
            trace.write(json.dumps({'t':step-1,'action':action,'before':before.tolist(),'after':after.tolist(),
                'rotation_xyzw_before':before_rotation,'target_fraction_after':fraction,
                'collided':bool(obs.get('collided',False)) if action==1 else False})+'\n')
            terminal=action==0 or step>=args.max_steps
            if terminal:
                final_distance=distance(after)
                success=int(action==0 and final_distance<.5 and fraction>=.025)
                record={'episode_id':e['episode_id'],'scene_id':scene,'category':e['category'],
                    'status':'complete','steps':step,'goal_stop':action==0,'success':success,
                    'spl':success*initial_distance/max(initial_distance,length,1e-8),
                    'initial_distance_m':initial_distance,'final_distance_m':final_distance,
                    'path_length_m':length,'target_fraction_at_end':fraction,
                    'category_supplied':args.category_assisted}
                records.append(record)
                write(out/(e['episode_id']+'.json'),record)
                write(out/'records.json',records)
                trace.close()
            transmit()
except Exception as exc:
    traceback.print_exc(file=sys.stderr)
    emit({'error':type(exc).__name__,'detail':str(exc)})
    raise
finally:
    if sim is not None:
        sim.close()
