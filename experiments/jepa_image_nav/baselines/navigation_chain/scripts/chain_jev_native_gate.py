"""Audit the native ROS build and run upstream graph fixtures, not navigation."""
import hashlib,json,os,signal,socket,subprocess,time,xmlrpc.client
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
prefix=ROOT/'runtime/ros_workspace/.pixi/envs/default'
workspace=ROOT/'repos/jev-objnav/jev_obj'
env=dict(os.environ,ROS_MASTER_URI='http://127.0.0.1:11321',ROS_IP='127.0.0.1',
    LD_LIBRARY_PATH=str(ROOT/'native_deps/lib')+':'+str(prefix/'lib'),
    ROS_PACKAGE_PATH=str(workspace/'src')+':'+str(prefix/'share'),
    ROS_ROOT=str(prefix/'share/ros'),ROS_DISTRO='noetic',ROS_VERSION='1',
    CMAKE_PREFIX_PATH=str(workspace/'devel')+':'+str(ROOT/'native_deps')+':'+str(prefix),
    PATH=str(prefix/'bin')+':'+os.environ['PATH'])
env.pop('LD_PRELOAD',None)
with socket.socket() as probe:
    assert probe.connect_ex(('127.0.0.1',11321))!=0,'Private fixture ROS port already occupied'
folder=ROOT/'results/jev_native_build_gate';folder.mkdir(parents=True,exist_ok=True)
with (folder/'roscore.log').open('w') as stream:
    master=subprocess.Popen([str(prefix/'bin/roscore'),'-p','11321'],env=env,stdout=stream,stderr=subprocess.STDOUT)
    try:
        for _ in range(30):
            if master.poll() is not None:raise RuntimeError('Fixture ROS master exited')
            try:
                code,_,_=xmlrpc.client.ServerProxy(env['ROS_MASTER_URI']).getUri('/fixture_gate')
                if code==1:break
            except OSError:pass
            time.sleep(1)
        else:raise TimeoutError('Fixture ROS master unavailable')
        subprocess.run([str(prefix/'bin/python'),'-c','import rospy,cv_bridge,em; assert hasattr(em,"RAW_OPT")'],env=env,check=True,timeout=15)
        fixtures=[]
        for name in ['region_graph2d_fixture','region_graph_dirty_fixture']:
            with (folder/(name+'.log')).open('w') as output:
                subprocess.run([str(workspace/'devel/lib/plan_env'/name)],env=env,check=True,
                    stdout=output,stderr=subprocess.STDOUT,timeout=30)
            fixtures.append(name)
        binaries={}
        for rel in ['exploration_manager/exploration_node','trajectory_manager/traj_server','lkh_mtsp_solver/tsp_node']:
            path=workspace/'devel/lib'/rel
            dependency=subprocess.run(['ldd',str(path)],env=env,capture_output=True,text=True,check=True)
            assert 'not found' not in dependency.stdout
            binaries[rel]={'sha256':hashlib.file_digest(path.open('rb'),'sha256').hexdigest(),'dynamic_dependencies_resolved':True}
        result={'passed':True,'fixtures':fixtures,'binaries':binaries,
            'source_commit':subprocess.check_output(['git','-C',str(ROOT/'repos/jev-objnav'),'rev-parse','HEAD'],text=True).strip(),
            'scope':'Native ROS message/graph fixtures and binary linkage only; NOT a closed-loop navigation performance gate',
            'ros_environment':'Isolated RoboStack Noetic; Python 3.12 / GCC 15',
            'host_ros_service_changed':False}
        (ROOT/'results/jev_native_build_gate.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(result,indent=2),flush=True)
    finally:
        master.send_signal(signal.SIGINT)
        try:master.wait(timeout=15)
        except subprocess.TimeoutExpired:
            master.terminate();master.wait(timeout=10)
