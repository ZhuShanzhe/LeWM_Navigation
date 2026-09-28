"""Frozen JEPLO ONNX student, deterministic headless MuJoCo evaluation.

Independent synchronous harness using the official robot, scan pattern, model,
observation ordering and PD interface. Not the unpublished paper benchmark runner.
Protocol differences are recorded explicitly; no training takes place.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
import argparse, json, math, time, hashlib, collections
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import mujoco as mj
import onnxruntime as ort

ROOT=Path(os.environ.get('JEPLO_WORKSPACE', '/root/autodl-tmp/jeplo_eval_20260928')).expanduser().resolve()
REPO=Path(os.environ.get('JEPLO_REPO', str(ROOT/'JEPLO'))).expanduser().resolve()
MODEL_DIR=REPO/'deployment/go2_deploy/model/pretrained'
ROBOT=REPO/'deployment/unitree_mujoco/unitree_robots/go2/go2.xml'
PATTERN=REPO/'training/go2_parkour/go2_parkour/sensor/scan_mode/mid360.npy'
NAMES=[leg+'_'+joint+'_joint' for joint in ['hip','thigh','calf'] for leg in ['FL','FR','RL','RR']]
Q0=np.array([.1,-.1,.1,-.1,.8,.8,1,1,-1.5,-1.5,-1.5,-1.5])

class Policy:
    def __init__(self, action_mode='deployment_default'):
        self.action_mode=action_mode
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self.est=ort.InferenceSession(str(MODEL_DIR/'sensor_estimator.onnx'),opts,providers=['CPUExecutionProvider'])
        self.pol=ort.InferenceSession(str(MODEL_DIR/'policy.onnx'),opts,providers=['CPUExecutionProvider'])
        self.reset()
    def reset(self):
        self.hist=np.zeros((1,450),np.float32);self.hidden=np.zeros((1,512),np.float32)
        self.feat=np.zeros((1,512),np.float32);self.action=np.zeros(12,np.float32)
        self.prev_depth=None;self.latencies=[];self.est_latencies=[]
    def step(self,obs,depth,k,reset_memory=False):
        self.hist[:,45:]=self.hist[:,:-45].copy();self.hist[:,:45]=obs
        if self.prev_depth is None:self.prev_depth=depth.copy()
        stack=np.stack([self.prev_depth,depth])[None].astype(np.float32)
        self.prev_depth=depth.copy()
        if k%5==0:
            if reset_memory:self.hidden[:]=0
            t=time.perf_counter()
            self.feat,self.hidden=self.est.run(['feature','hidden_out'],{'prop_hist':self.hist,'depth_stack':stack,'hidden_in':self.hidden})
            self.est_latencies.append((time.perf_counter()-t)*1000)
        t=time.perf_counter()
        previous=self.action.copy()
        self.action=self.pol.run(['actions'],{'proprio_hist':self.hist,'high_feat':self.feat})[0][0]
        self.latencies.append((time.perf_counter()-t)*1000)
        if not np.all(np.isfinite(self.action)):raise RuntimeError('Nonfinite model action')
        applied=self.action.copy()
        if self.action_mode=='deployment_default':
            applied=.8*applied+.2*previous
            applied[:4]*=.5
        return Q0+.25*applied

class Lidar:
    def __init__(self,model,scan_phase=0,accumulate=5):
        angles=np.load(PATTERN);self.frames=[]
        n=len(angles);count=n//math.gcd(n,24000)
        for frame in range(count):
            a=angles[(frame*24000+np.arange(24000))%n].astype(np.float64)
            phi=(a[:,0]+np.pi)%(2*np.pi)-np.pi;theta=a[:,1]
            # std::round compatibility, rather than numpy banker's rounding.
            rr=(theta/np.pi*180+4.5)/2;cc=(phi/np.pi*180+120)/2
            row=(np.sign(rr)*np.floor(np.abs(rr)+.5)).astype(int)
            col=(np.sign(cc)*np.floor(np.abs(cc)+.5)).astype(int)
            # Official publisher defaults to downsample_rate=1; do not silently thin rays.
            indices=np.flatnonzero((row>=0)&(row<25)&(col>=0)&(col<120))
            narrow_col=col[indices]-30
            indices=indices[(narrow_col>=0)&(narrow_col<60)]
            phi,theta=phi[indices],theta[indices]
            dirs=np.stack([np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)],-1)
            bins=row[indices]*60+col[indices]-30
            self.frames.append((dirs,bins))
        self.frame=scan_phase%len(self.frames)
        self.site=mj.mj_name2id(model,mj.mjtObj.mjOBJ_SITE,'lidar')
        self.body=mj.mj_name2id(model,mj.mjtObj.mjOBJ_BODY,'base_link')
        self.hist=collections.deque([np.ones(1500,np.float32) for _ in range(accumulate)],maxlen=accumulate)
        self.groups=np.array([1,1,1,0,0,0],np.uint8)
        self.latest=np.ones((25,60),np.float32)
    def scan(self,m,d,rng,noise=0.):
        dirs,bins=self.frames[self.frame];self.frame=(self.frame+1)%len(self.frames)
        world=np.ascontiguousarray(dirs@d.site_xmat[self.site].reshape(3,3).T)
        dist=np.empty(len(dirs),np.float64);ids=np.empty(len(dirs),np.int32)
        mj.mj_multiRay(m,d,d.site_xpos[self.site],world.reshape(-1),self.groups,1,self.body,ids,dist,len(dirs),100.)
        valid=(dist>.1)&(dist<=2)&np.isfinite(dist)
        ranges=dist[valid]
        if noise:ranges=ranges+rng.normal(0,noise,len(ranges))
        out=np.ones(1500,np.float32)
        np.minimum.at(out,bins[valid],np.clip(ranges/2,0,1).astype(np.float32))
        self.hist.append(out)
    def publish(self):
        self.latest=np.min(np.stack(self.hist),axis=0).reshape(25,60)
        return self.latest.copy()

def mask_for(condition,rng):
    yy,xx=np.mgrid[:25,:60];mask=np.zeros((25,60),bool)
    if condition.startswith('mask'):
        mask.ravel()[rng.choice(1500,round(1500*float(condition[4:])),replace=False)]=True
    elif condition=='cut':mask[12:,:]=True
    elif condition=='cross':mask=(np.abs(xx-30)<3)|(np.abs(yy-12)<2)
    elif condition=='stripe':mask=(np.abs(xx-7)<2)|(np.abs(xx-52)<2)
    elif condition=='blob':
        for _ in range(3):
            x,y=rng.uniform(5,55),rng.uniform(4,21)
            mask|=((xx-x)/6)**2+((yy-y)/5)**2<1
    elif condition=='blind':mask[:]=True
    return mask

def make_scene(kind,height):
    include=str(ROBOT)
    terrain='<geom name="floor" type="plane" size="0 0 .05" friction="2 .005 .0001" rgba=".28 .32 .38 1"/>'
    if kind=='box':
        terrain+=f'<geom name="obstacle_box" type="box" pos="1.75 0 {height/2}" size=".75 1.5 {height/2}" rgba=".55 .62 .72 1"/>'
        goal=1.55;top=height
    elif kind=='stairs':
        for j in range(6):
            h=(j+1)*height
            terrain+=f'<geom name="stair_{j}" type="box" pos="{1+.3*j+.15} 0 {h/2}" size=".15 2 {h/2}" rgba=".55 .62 .72 1"/>'
        terrain+=f'<geom name="top" type="box" pos="3.8 0 {3*height}" size="1 2 {3*height}" rgba=".55 .62 .72 1"/>'
        goal=3.;top=height*6
    elif kind=='flat':goal=4.;top=0
    else:raise ValueError(kind)
    root=ET.parse(ROBOT).getroot()
    root.find('compiler').set('meshdir',str(ROBOT.parent/'assets'))
    root.find('option').set('timestep','.002')
    world=root.find('worldbody')
    for item in ET.fromstring(f'<worldbody><light pos="0 0 6" dir="0 0 -1"/>{terrain}</worldbody>'):
        world.append(item)
    return ET.tostring(root,encoding='unicode'),goal,top

def run(case,policy,save_trace=False,video_path=None):
    rng=np.random.default_rng(case['seed'])
    if 'scene_file' in case:
        xml=Path(case['scene_file']).read_text();goal=case['goal_x'];top=case['goal_z']
    else:xml,goal,top=make_scene(case['terrain'],case['height'])
    m=mj.MjModel.from_xml_string(xml);d=mj.MjData(m)
    qids=np.array([m.jnt_qposadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)] for n in NAMES])
    vids=np.array([m.jnt_dofadr[mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n)] for n in NAMES])
    aids=np.array([np.flatnonzero(m.actuator_trnid[:,0]==mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,n))[0] for n in NAMES])
    bid=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,'base_link')
    gyro=mj.mj_name2id(m,mj.mjtObj.mjOBJ_SENSOR,'imu_gyro');ga=m.sensor_adr[gyro]
    condition=case['condition']
    if condition=='friction_low':
        # Set both feet and terrain, since MuJoCo contact priority can select feet.
        m.geom_friction[:,0]=.15
    if condition=='mass_plus20':
        # Uniform whole-body mass/inertia scaling, NOT a physical backpack model.
        m.body_mass[1:]*=1.2;m.body_inertia[1:]*=1.2;mj.mj_setConst(m,d)
    yaw=case['yaw'];d.qpos[:3]=[case['x'],case['y'],case.get('spawn_z',.38)];d.qpos[3:7]=[math.cos(yaw/2),0,0,math.sin(yaw/2)]
    d.qpos[qids]=Q0;mj.mj_forward(m,d)
    # Standing startup, identical for all conditions; no policy selection/tuning.
    for _ in range(250):
        d.ctrl[aids]=28*(Q0-d.qpos[qids])-1.1*d.qvel[vids];mj.mj_step(m,d)
    d.time=0;policy.reset()
    condition=case['condition'];acc=1 if condition=='sparse1' else 5
    lidar=Lidar(m,case['scan_phase'],acc)
    mask=mask_for(condition,np.random.default_rng(case['seed']+100000))
    # Deployment waits for a first frame. Prime one physical scan at the standing pose.
    lidar.scan(m,d,rng);depth=lidar.publish();depthqueue=collections.deque()
    delay=.3 if condition=='delay300' else (.1 if condition=='delay100' else 0)
    noise=.05 if condition=='noise5cm' else 0
    trace=[];errxy=[];erryaw=[];energy=0;max_tau=0.;distance=0;contact_steps=0;fallen=0;goal_ticks=0
    lastpos=d.qpos[:2].copy();target=Q0.copy();success=False;reason='timeout'
    depth_coverage=[];start=time.perf_counter();next_scan=.05;frames=[];renderer=None
    limits=np.array([[-1.0472,1.0472]]*4+[[-1.5708,3.4907]]*2+[[-.5236,4.5379]]*2+[[-2.7227,-.8378]]*4)
    widths=limits[:,1]-limits[:,0];limits[:,0]+=.05*widths;limits[:,1]-=.05*widths
    widths=limits[:,1]-limits[:,0];lower=limits[:,0]+.1*widths;upper=limits[:,1]-.1*widths
    ood_count=0;guard_count=0;max_proxy=0.;max_joint_vel=0.;stop_speeds=[];stop_positions=[];tracking_time=0.
    policy.action_mode=case.get('action_mode','deployment_default')
    if video_path:
        import imageio.v2 as imageio
        renderer=mj.Renderer(m,height=480,width=640)
        cam=mj.MjvCamera();cam.distance=5;cam.azimuth=130;cam.elevation=-20
    maxsteps=int(case['duration']/.02)
    for k in range(maxsteps):
        R=d.xmat[bid].reshape(3,3);yawnow=math.atan2(R[1,0],R[0,0])
        yawerr=(yaw-yawnow+np.pi)%(2*np.pi)-np.pi
        speed=case['speed'] if k>=25 else 0.
        if condition=='speed_slow':speed*=.5
        if condition=='speed_fast':speed*=1.5
        cmd=np.array([speed*math.cos(yawnow),-speed*math.sin(yawnow),np.clip(yawerr,-1,1)])
        if 'command_segments' in case:
            cmd=np.zeros(3)
            for segment in case['command_segments']:
                if segment['start']<=d.time<segment['end']:
                    cmd=np.array(segment['cmd'],dtype=float);break
        cmd=np.clip(cmd,[-1.5,-1.5,-3.],[1.5,1.5,3.])
        obs=np.concatenate([d.sensordata[ga:ga+3]*.25,R.T@np.array([0,0,-1]),cmd*np.array([2,2,.25]),d.qpos[qids]-Q0,d.qvel[vids]*.05,policy.action*.1]).astype(np.float32)
        depth_input=depth.copy();depth_input[mask]=1
        if condition=='blackout1s' and 1.5<=d.time<2.5:depth_input[:]=1
        target=policy.step(np.clip(obs,-100,100),depth_input,k,condition=='reset_memory')
        q=d.qpos[qids]
        guarded=((q<=lower)&(target<q))|((q>=upper)&(target>q))
        target[guarded]=q[guarded];guard_count+=int(guarded.sum())
        proxy=float(np.max(np.abs(28*(target-q)-1.1*d.qvel[vids])))
        joint_vel=float(np.max(np.abs(d.qvel[vids])))
        max_proxy=max(max_proxy,proxy);max_joint_vel=max(max_joint_vel,joint_vel)
        ood_count=ood_count+1 if proxy>30 or joint_vel>25 else 0
        if ood_count>=20:
            reason='deployment_safety_stop';break
        velocity=R.T@d.qvel[:3]
        if k>=25:
            errxy.append(float(np.sum((velocity[:2]-cmd[:2])**2)))
            erryaw.append(float((d.sensordata[ga+2]-cmd[2])**2))
            tracking_time+=.02
        if 'stop_at' in case and d.time>=case['stop_at']:
            stop_speeds.append(float(np.linalg.norm(velocity[:2])));stop_positions.append(d.qpos[:2].copy())
        for _ in range(10):
            d.xfrc_applied[bid,:]=0
            if condition=='push_lateral' and 2.<=d.time<2.2:d.xfrc_applied[bid,1]=60.
            tau=28*(target-d.qpos[qids])-1.1*d.qvel[vids]
            d.ctrl[aids]=tau;mj.mj_step(m,d)
            if d.time+1e-9>=next_scan:
                lidar.scan(m,d,rng,noise);next_scan+=.05
            energy+=float(np.sum(np.abs(d.actuator_force[aids]*d.qvel[vids])))*.002
            max_tau=max(max_tau,float(np.max(np.abs(d.actuator_force))))
        if (k+1)%5==0:
            fresh=lidar.publish();depthqueue.append((d.time+delay,fresh))
            depth_coverage.append(float(np.mean(fresh<1)))
        while depthqueue and depthqueue[0][0]<=d.time+1e-8:_,depth=depthqueue.popleft()
        pos=d.qpos[:3].copy();distance+=float(np.linalg.norm(pos[:2]-lastpos));lastpos=pos[:2].copy()
        upright=d.xmat[bid].reshape(3,3)[2,2]
        bodycontact=False
        for c in d.contact:
            ba,bb=m.geom_bodyid[c.geom1],m.geom_bodyid[c.geom2]
            if (ba==bid and bb==0) or (bb==bid and ba==0):bodycontact=True;break
        contact_steps+=int(bodycontact)
        fallen=fallen+1 if upright<.2 or pos[2]<.13 else 0
        ongoal=pos[0]>=goal and abs(pos[1])<1.2 and pos[2]>top+.17 and upright>.6
        goal_ticks=goal_ticks+1 if ongoal else 0
        if save_trace:trace.append({'t':round(d.time,3),'pos':pos.tolist(),'upright':float(upright),'action':policy.action.tolist(),'cmd':cmd.tolist(),'body_velocity':velocity.tolist(),'tau_proxy':proxy,'coverage':float(np.mean(depth_input<1))})
        if renderer and k%2==0:
            cam.lookat[:]=pos;renderer.update_scene(d,camera=cam);frames.append(renderer.render().copy())
        if not np.all(np.isfinite(d.qpos)):reason='numerical_failure';break
        if fallen>=10:reason='fall';break
        if case.get('task')!='command_tracking' and (abs(pos[1])>2.5 or pos[0]<-2):reason='out_of_bounds';break
        if case.get('task')!='command_tracking' and goal_ticks>=25:success=True;reason='success';break
    if case.get('task')=='command_tracking' and reason=='timeout':
        success=True;reason='survived_command_sequence'
    elapsed=time.perf_counter()-start
    if renderer:
        imageio.mimsave(video_path,frames,fps=25);renderer.close()
    result={**case,'success':success,'reason':reason,'sim_seconds':float(d.time),'wall_seconds':elapsed,'position':d.qpos[:3].tolist(),'progress_x':float(d.qpos[0]-case['x']),'path_length':distance,'tracking_xy_mse':float(np.mean(errxy)) if errxy else None,'tracking_yaw_mse':float(np.mean(erryaw)) if erryaw else None,'mechanical_work_abs_J':energy,'max_abs_torque_Nm':max_tau,'base_contact_step_fraction':contact_steps/(k+1),'mask_fraction':float(mask.mean()),'scan_coverage':float(np.mean(depth_coverage)) if depth_coverage else 0,'policy_ms_p50':float(np.median(policy.latencies)),'policy_ms_p95':float(np.percentile(policy.latencies,95)),'estimator_ms_p95':float(np.percentile(policy.est_latencies,95)),'runtime':'onnxruntime_cpu_sync','protocol':case.get('protocol','independent_mujoco_v1'),'policy_sha256':hashlib.sha256((MODEL_DIR/'policy.onnx').read_bytes()).hexdigest()}
    result.update(protocol='deployment_interface_v2',action_mode=policy.action_mode,tracking_xy_rmse=float(np.sqrt(np.mean(errxy))) if errxy else None,tracking_yaw_rmse=float(np.sqrt(np.mean(erryaw))) if erryaw else None,tracking_time_s=tracking_time,joint_guard_events=guard_count,max_tau_proxy_Nm=max_proxy,max_joint_velocity_rad_s=max_joint_vel,stop_final_speed_m_s=float(np.mean(stop_speeds[-25:])) if stop_speeds else None,stop_displacement_m=float(np.linalg.norm(stop_positions[-1]-stop_positions[0])) if stop_positions else None)
    return result,trace

def cases(args):
    terrains=args.terrains.split(',');conditions=args.conditions.split(',')
    items=[]
    for terrain in terrains:
        height={'flat':0.,'box':.5,'stairs':.2}[terrain]
        for index in range(args.trials):
            seed=args.seed+index;r=np.random.default_rng(seed)
            yaw=r.uniform(-np.pi,np.pi) if terrain=='stairs' else r.uniform(-.26,.26)
            base={'terrain':terrain,'height':height,'seed':seed,'yaw':float(yaw),'x':float(r.uniform(-.15,.15)),'y':float(r.uniform(-.15,.15)),'speed':.75,'scan_phase':int(r.integers(0,1000)),'duration':args.duration,'split':args.split}
            for cond in conditions:items.append({**base,'condition':cond})
    return items

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--terrains',default='flat');parser.add_argument('--conditions',default='clean');parser.add_argument('--trials',type=int,default=1);parser.add_argument('--seed',type=int,default=100);parser.add_argument('--duration',type=float,default=20);parser.add_argument('--output',required=True);parser.add_argument('--split',default='development');parser.add_argument('--trace',action='store_true');parser.add_argument('--video',action='store_true')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    suite=cases(args);(out/'episodes.json').write_text(json.dumps(suite,indent=2));policy=Policy()
    for i,case in enumerate(suite):
        result,trace=run(case,policy,args.trace,str(out/f'episode_{i:05d}.mp4') if args.video else None)
        with (out/'results.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
        if trace:(out/f'trace_{i:05d}.json').write_text(json.dumps(trace))
        print(json.dumps({'episode':i,**result}),flush=True)
    (out/'complete').write_text('completed\n')

if __name__=='__main__':main()
