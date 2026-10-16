"""Loopback-only native-stack smoke test. No physical robot access.

Runs the recommended raw-action/stride-4/cage profile and records asynchronous
LiDAR sensor poses. It does not implement paper success criteria.
"""
import argparse,json,os,signal,struct,subprocess,time,shutil,hashlib
import socket
from pathlib import Path
import numpy as np
import zmq
import xml.etree.ElementTree as ET
p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--duration',type=float,default=10);p.add_argument('--case',choices=['stand','forward','rotate','stairs','boxes'],default='stand');p.add_argument('--stop-at-ascent-goal',action='store_true');a=p.parse_args()
if a.stop_at_ascent_goal and a.case not in ['stairs','boxes']:p.error('Ascent goal is only defined for stairs/boxes')
root=a.root.resolve();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
shutil.copy2(Path(__file__),out/'runner_snapshot.py')
manifest=json.loads((root/'native_build_manifest.json').read_text())
for rel,expected in manifest['artifacts'].items():
    with (root/rel).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError(f'Native build artifact drift: {rel}')
profile=json.loads((root/'official_readme_profile.json').read_text())
for name,expected in profile['models_sha256'].items():
    with (root/'source'/profile['model_subdir']/name).open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
    if actual!=expected:raise RuntimeError(f'Checkpoint drift: {name}')
if Path('/tmp/.X11-unix/X97').exists():raise RuntimeError('Display :97 is already in use')
for port in [5560,5562,5590]:
    with socket.socket() as sock:
        # Match ZeroMQ's address reuse: TIME_WAIT is not an active foreign service.
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
        sock.bind(('127.0.0.1',port))
env=os.environ.copy();prefix=root/'dependencies/prefix/usr';capi=root/'ort_env/lib/python3.11/site-packages/onnxruntime/capi'
env['LD_LIBRARY_PATH']=':'.join(map(str,[prefix/'lib/x86_64-linux-gnu',prefix/'local/lib',root/'dependencies/mujoco-3.3.7/lib',capi,root/'runtime_env/lib/python3.11/site-packages/tensorrt_libs',Path('/usr/local/cuda-12.8/lib64'),Path('/usr/lib/x86_64-linux-gnu'),root.parent/'gl_libs/extracted/usr/lib/x86_64-linux-gnu']))
env['DISPLAY']=':97';env['PYTHONDONTWRITEBYTECODE']='1'
processes=[];handles=[];ctx=zmq.Context();pub=ctx.socket(zmq.PUB);sub=ctx.socket(zmq.SUB)
pub.setsockopt(zmq.LINGER,0);sub.setsockopt(zmq.LINGER,0);sub.setsockopt(zmq.SUBSCRIBE,b'')
pub.bind('tcp://127.0.0.1:5562');sub.connect('tcp://127.0.0.1:5590')
def launch(name,args,cwd=None,stdin=None):
    handle=(out/(name+'.log')).open('w');handles.append(handle)
    proc=subprocess.Popen(['stdbuf','-oL','-eL',*args],cwd=cwd,env=env,stdout=handle,stderr=subprocess.STDOUT,stdin=stdin,start_new_session=True)
    processes.append((name,proc));return proc
records=[];report={'scope':'native-stack deployment README smoke; self-defined commands, not paper performance','case':a.case,'complete':False,'requested_duration':a.duration,'command':{'stand':[0,0,0],'forward':[.25,0,0],'rotate':[0,0,1],'stairs':[.75,0,0],'boxes':[.75,0,0]}[a.case],'stop_at_ascent_goal':a.stop_at_ascent_goal}
goal_time=None;start_base=None;start_heading=None;held_upright=True
try:
    launch('display',[str(prefix/'bin/Xvfb'),':97','-screen','0','1280x720x24','-nolisten','tcp'])
    time.sleep(1)
    simdir=root/'source/deployment/unitree_mujoco/simulate/build'
    simulator=launch('simulator',[str(simdir/'unitree_mujoco'),'--lidar','--network','lo','--domain_id','1'],cwd=simdir)
    launch('sensor',[str(root/'lidar_depth_pub_sim'),'--sim','--downsample-rate','4','--fov','25x60'])
    controller=launch('controller',[str(root/'controller_output/deploy'),'--net','lo','--model-dir','pretrained','--ood-count-threshold','15','--raw-actions'],cwd=root/'controller_output',stdin=subprocess.PIPE)
    controller.stdin.write(b'\n');controller.stdin.flush()
    start=time.monotonic();ready=None;last_cmd=0
    while time.monotonic()-start<90:
        now=time.monotonic()
        if now-last_cmd>=.05:
            cmd=report['command'] if ready is not None and now-ready>2 and goal_time is None else [0,0,0]
            pub.send(struct.pack('<3f',*cmd));last_cmd=now
        if sub.poll(timeout=5):
            raw=sub.recv()
            if len(raw)>=48:
                pose=np.frombuffer(raw[:48],dtype='<f4').copy()
                records.append({'wall_seconds':now-start,'sensor_xyz':pose[:3].tolist(),'sensor_rotation':pose[3:].tolist(),'points':(len(raw)-48)//12})
                body_r=pose[3:].reshape(3,3)@np.diag([1,-1,-1]);base=pose[:3]-body_r@np.array([.275,0,.149])
                if ready is not None and start_base is None:
                    start_base=base.copy();start_heading=body_r[:2,0]/np.linalg.norm(body_r[:2,0])
                if a.stop_at_ascent_goal and start_base is not None:
                    forward=float((base[:2]-start_base[:2])@start_heading)
                    gx,gz=(4.15,.9) if a.case=='boxes' else (8.05,3.15)
                    if goal_time is None and forward>=gx and base[2]>=gz and body_r[2,2]>.8:
                        goal_time=now
                        report['ascent_goal']={'definition':'self-defined demo stopping gate, NOT paper criterion','forward_displacement_min':gx,'base_z_min':gz,'upright_cos_min':.8,'hold_seconds':2,'reached_seconds':now-start}
                    if goal_time is not None:held_upright=held_upright and body_r[2,2]>.8 and base[2]>=gz
        for name,proc in processes:
            if proc.poll() is not None:raise RuntimeError(f'{name} exited with {proc.returncode}')
        log=(out/'controller.log').read_text(errors='replace')
        if ready is None and 'Locomotion control sequence started, robot ready' in log:
            if a.case=='boxes':
                # Use the original public Ctrl+2 interface; no scene/physics mutation by this harness.
                xdotool=str(prefix/'bin/xdotool')
                result=subprocess.run([xdotool,'search','--onlyvisible','--pid',str(simulator.pid)],env=env,capture_output=True,text=True,check=True,timeout=5)
                windows=result.stdout.split();assert len(windows)==1,windows
                subprocess.run([xdotool,'windowfocus','--sync',windows[0]],env=env,check=True,timeout=5)
                time.sleep(.15)
                subprocess.run([xdotool,'key','--clearmodifiers','--delay','100','ctrl+2'],env=env,check=True,timeout=5)
                ack_deadline=time.monotonic()+3
                while time.monotonic()<ack_deadline:
                    if 'active terrain is boxes' in (out/'simulator.log').read_text(errors='replace'):break
                    pub.send(struct.pack('<3f',0,0,0));time.sleep(.05)
                assert 'active terrain is boxes' in (out/'simulator.log').read_text(errors='replace'),'Terrain switch not acknowledged'
                report['terrain_switch']='official Ctrl+2 acknowledged'
            ready=time.monotonic()
            report['ready_seconds']=ready-start
        if ready is not None and now-ready>=a.duration:
            report['complete']=True;break
        if goal_time is not None and now-goal_time>=2:
            report['complete']=True;report['ascent_goal_held']=bool(held_upright);break
    report['ready_seconds']=None if ready is None else ready-start
    report['received_lidar_packets']=len(records)
    if not report['complete']:raise RuntimeError('Native controller did not complete ready-state smoke deadline')
    samples=[r for r in records if r['wall_seconds']>report['ready_seconds']+2]
    if len(samples)>10:
        xyz=np.array([r['sensor_xyz'] for r in samples]);t=np.array([r['wall_seconds'] for r in samples])
        rot=np.array([r['sensor_rotation'] for r in samples]).reshape(-1,3,3)
        xml=root/'source/deployment/unitree_mujoco/unitree_robots/go2/go2.xml'
        site=ET.parse(xml).find('.//site[@name="lidar"]')
        assert site is not None and np.array_equal(np.fromstring(site.get('quat'),sep=' '),[0,1,0,0])
        offset=np.fromstring(site.get('pos'),sep=' ')
        body_rot=rot@np.diag([1,-1,-1]);body_xyz=xyz-np.einsum('nij,j->ni',body_rot,offset)
        yaw=np.unwrap(np.arctan2(body_rot[:,1,0],body_rot[:,0,0]));elapsed=t[-1]-t[0]
        report['pose_metrics']={'measurement':'base-link pose reconstructed from official LiDAR site extrinsics','samples':len(samples),'measured_lidar_hz':float((len(t)-1)/elapsed),'measurement_seconds':float(elapsed),'xyz_first':body_xyz[0].tolist(),'xyz_last':body_xyz[-1].tolist(),'xy_displacement_m':float(np.linalg.norm(body_xyz[-1,:2]-body_xyz[0,:2])),'yaw_change_rad':float(yaw[-1]-yaw[0]),'mean_yaw_rate_rad_s':float((yaw[-1]-yaw[0])/elapsed),'mean_xy_displacement_speed_m_s':float(np.linalg.norm(body_xyz[-1,:2]-body_xyz[0,:2])/elapsed),'min_base_z':float(body_xyz[:,2].min()),'max_base_z':float(body_xyz[:,2].max()),'min_base_upright_cos':float(body_rot[:,2,2].min()),'max_base_x':float(body_xyz[:,0].max())}
    report['ood_positive_log_lines']=sum('SAFETY ood=1' in line for line in log.splitlines())
    report['safety_stop_detected']=('safety fault' in log.lower() or 'OOD safety triggered' in log)
except Exception as exc:
    report['error']=str(exc)
finally:
    pub.send(struct.pack('<3f',0,0,0))
    for name,proc in reversed(processes):
        if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
    for name,proc in reversed(processes):
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    for h in handles:h.close()
    pub.close();sub.close();ctx.term()
    (out/'lidar_pose.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
raise SystemExit(0 if report['complete'] else 2)
