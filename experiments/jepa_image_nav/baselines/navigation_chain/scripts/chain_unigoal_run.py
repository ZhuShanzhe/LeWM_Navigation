"""Original UniGoal main/agent/graph/BEV/FMM with audited simulator transport.

Published as an adapted cross-domain run, NOT a paper-number reproduction.
The native source is not replaced by a handcrafted frontier policy.
"""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys
import tempfile
import traceback
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from chain_unigoal_import import ROOT, REPO, prepare

class SensorTransport:
    def __init__(self,panel,out,category_assisted,max_steps):
        self.temporary=tempfile.TemporaryDirectory(prefix='unigoal_',dir='/dev/shm')
        self.buffer=Path(self.temporary.name)/'sensors.npz'
        self.log=(out/'simulator_stderr.log').open('w')
        command=['/root/autodl-tmp/jepa_vln_data_20261014/env/bin/python',
            str(ROOT/'tools/chain_unigoal_sim.py'),'--panel',str(panel),
            '--output',str(out),'--buffer',str(self.buffer),'--max-steps',str(max_steps)]
        if category_assisted:
            command+=['--category-assisted']
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
            stderr=self.log,text=True,bufsize=1,
            env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
        self.poll=selectors.DefaultSelector()
        self.poll.register(self.process.stdout,selectors.EVENT_READ)
    def call(self,request):
        self.process.stdin.write(json.dumps(request)+'\n')
        self.process.stdin.flush()
        if not self.poll.select(120):
            raise TimeoutError('Habitat sensor process did not respond')
        line=self.process.stdout.readline()
        assert line,'Habitat sensor process exited'
        result=json.loads(line)
        assert 'error' not in result,result
        value=np.load(result['buffer'])
        assert set(value.files)=={'rgb','depth','gps','compass','position','rotation_xyzw','goal_rgb'}
        self.data={key:value[key].copy() for key in value.files}
        value.close()
        return result
    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"op":"close"}\n')
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        self.poll.close()
        self.log.close()
        self.temporary.cleanup()

def select_panel(limit,out):
    source=Path('/root/autodl-tmp/jepa_vln_baselines_20261015/plans/hssd_image_nav_v1_test_instance_20perscene.jsonl')
    episodes=[json.loads(x) for x in source.read_text().splitlines()]
    if limit:
        selected=[]
        # Deterministic breadth-first coverage across scenes, independent of all
        # model outcomes. Hash order avoids selecting only the first easy goals.
        scenes=sorted({e['scene_id'] for e in episodes})
        buckets={s:sorted([e for e in episodes if e['scene_id']==s],
            key=lambda e:hashlib.sha256(e['episode_id'].encode()).hexdigest()) for s in scenes}
        for rank in range(20):
            for scene in scenes:
                selected.append(buckets[scene][rank])
        episodes=selected[:limit]
    path=out/'panel.jsonl'
    path.write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    return path,episodes

def execute(options):
    out=ROOT/'results'/options.run
    out.mkdir(parents=True,exist_ok=True)
    panel,episodes=select_panel(options.limit,out)
    transport=SensorTransport(panel,out,options.category_assisted,options.max_steps)
    try:
        if options.sim_gate:
            initial=transport.call({'op':'reset'})
            first={k:v.copy() for k,v in transport.data.items()}
            assert first['rgb'].shape==(640,480,3) and first['depth'].shape==(640,480,1)
            assert first['depth'].min()>=0 and first['depth'].max()<=1
            assert np.max(np.abs(first['gps']))<1e-5 and abs(first['compass'][0])<1e-5
            transport.call({'op':'step','action':2})
            second=transport.data
            assert np.linalg.norm(second['position']-first['position'])<1e-5
            assert abs(abs(float(second['compass'][0]))-math.radians(15))<1e-5
            result={'passed':True,'rgb_shape':list(first['rgb'].shape),'depth_shape':list(first['depth'].shape),
                'depth_range':[float(first['depth'].min()),float(first['depth'].max())],
                'zero_relative_pose':True,'turn_angle_degrees':15,'depth_max_m':30,
                'goal_label_or_semantic_image_transmitted':False,
                'geometry':'Frozen common HSSD body/navmesh; native UniGoal camera and 15-degree actions'}
            (out/'simulator_gate.json').write_text(json.dumps(result,indent=2))
            print(json.dumps(result),flush=True)
            return
        Graph,Agent,BEV=prepare()
        assert (ROOT/'results/bert_assets.json').exists(), 'Prepare BERT assets before acquiring the GPU'
        from groundingdino import _C
        from groundingdino.models.GroundingDINO.ms_deform_attn import multi_scale_deformable_attn_pytorch
        torch.manual_seed(20261016)
        value=torch.randn(1,4,2,4,device='cuda')
        shapes=torch.tensor([[2,2]],dtype=torch.long,device='cuda')
        starts=torch.tensor([0],dtype=torch.long,device='cuda')
        locations=torch.rand(1,3,2,1,2,2,device='cuda')
        attention=torch.rand(1,3,2,1,2,device='cuda')
        attention=(attention/attention.sum((-1,-2),keepdim=True)).contiguous()
        actual=_C.ms_deform_attn_forward(value,shapes,starts,locations,attention,64)
        reference=multi_scale_deformable_attn_pytorch(value,shapes,locations,attention)
        maximum_error=float((actual-reference).abs().max())
        assert maximum_error<1e-5
        (out/'cuda_operator_gate.json').write_text(json.dumps({'passed':True,
            'native_cuda_kernel_vs_publisher_pytorch_reference_max_error':maximum_error,
            'compatibility_patch':'Two AT_DISPATCH type arguments: value.type() to value.scalar_type()',
            'device':torch.cuda.get_device_name()},indent=2))
        from groundingdino.util import get_tokenlizer
        native_tokenizer=get_tokenlizer.get_tokenlizer
        native_text_model=get_tokenlizer.get_pretrained_language_model
        get_tokenlizer.get_tokenlizer=lambda name:native_tokenizer(str(ROOT/'models/bert-base-uncased') if name=='bert-base-uncased' else name)
        get_tokenlizer.get_pretrained_language_model=lambda name:native_text_model(str(ROOT/'models/bert-base-uncased') if name=='bert-base-uncased' else name)
        from src.envs.instanceimagegoal_env import InstanceImageGoal_Env
        from src.utils.camera import get_camera_matrix
        from configs.categories import name2index
        import src.envs
        import src.utils.visualization.semantic_prediction as semantic
        from detectron2.checkpoint import DetectionCheckpointer
        native_load=DetectionCheckpointer.load
        native_load_model=DetectionCheckpointer._load_model
        def safe_detector_load(self,path,*args,**kwargs):
            if path=='detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl':
                path=str(ROOT/'weights/mask_rcnn_R_50_FPN_3x.safe.pth')
            return native_load(self,path,*args,**kwargs)
        def audited_detector_model(self,checkpoint):
            result=native_load_model(self,checkpoint)
            assert not result.unexpected_keys and not result.missing_keys,result
            assert not result.incorrect_shapes,result
            (out/'detector_weight_gate.json').write_text(json.dumps({'passed':True,
                'missing_keys':list(result.missing_keys),'unexpected_keys':list(result.unexpected_keys),
                'incorrect_shapes':list(result.incorrect_shapes)},indent=2))
            return result
        DetectionCheckpointer.load=safe_detector_load
        DetectionCheckpointer._load_model=audited_detector_model
        class RPCEnvironment(InstanceImageGoal_Env):
            def __init__(self,args):
                self.args=args
                self.name2index=name2index
                self.index2name={v:k for k,v in name2index.items()}
                self.info={}
                self.episode_no=-1
                self.rank=0
                self.was_done=False
            def observation(self):
                return {k:transport.data[k] for k in ['rgb','depth','gps','compass']}
            def get_agent_pose(self):
                import quaternion
                data=transport.data
                q=data['rotation_xyzw']
                return {'position':data['position'],'rotation':quaternion.quaternion(q[3],*q[:3])}
            def reset(self):
                previous={k:self.info.get(k) for k in ['success','spl','distance_to_goal','soft_spl']}
                reply=transport.call({'op':'reset'})
                self.info=reply['info']
                if self.was_done:
                    self.info.update(previous)
                self.was_done=False
                self.episode_no=self.info['episode_no']
                self.instance_imagegoal=transport.data['goal_rgb']
                supplied=reply.get('supplied_category')
                native_name={'couch':'sofa','tv':'tv_monitor'}.get(supplied,supplied)
                self.goal_name=native_name or 'unknown'
                self.gt_goal_idx=self.name2index[native_name] if native_name else -1
                self.info.update(goal_name=self.goal_name,goal_cat_id=self.gt_goal_idx,
                    instance_imagegoal=self.instance_imagegoal,sensor_pose=[0.,0.,0.])
                self.last_agent_location=self.get_agent_location()
                return self.observation(),self.info
            def step(self,action):
                assert set(action)=={'action'}
                reply=transport.call({'op':'step','action':int(action['action'])})
                updated=reply['info']
                for k,v in self.info.items():
                    if k not in updated:
                        updated[k]=v
                self.info=updated
                self.info['sensor_pose']=list(self.get_location_change())
                self.was_done=reply['done']
                return self.observation(),reply['done'],self.info
        class CalibratedGraph(Graph):
            def __init__(self,args,*a,**kw):
                super().__init__(args,*a,**kw)
                self.camera_matrix=get_camera_matrix(args.env_frame_width,args.env_frame_height,args.hfov)
            def set_observations(self,observations):
                super().set_observations(observations)
                self.image_depth=self.image_depth*30.
        spec=importlib.util.spec_from_file_location('publisher_unigoal_main',REPO/'main.py')
        entry=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(entry)
        config=yaml.safe_load((REPO/'configs/config_habitat.yaml').read_text())
        config.update(goal_type='ins-image',episode_id=-1,goal='',real_world=False,visualize=0,
            dump_location=str(out),experiment_id='native_inference',num_eval_episodes=len(episodes),
            max_episode_length=options.max_steps,sim_gpu_id=0,
            base_url=options.base_url,
            llm_model='llama3.2-vision:11b',vlm_model='llama3.2-vision:11b')
        config.update(map_size=config['map_size_cm']//config['map_resolution'],is_debugging=False)
        config.update(global_width=config['map_size'],global_height=config['map_size'],
            local_width=config['map_size']//config['global_downscaling'],
            local_height=config['map_size']//config['global_downscaling'],device=torch.device('cuda:0'),
            num_scenes=1,num_episodes=len(episodes),log_dir=str(out/'native_logs'),visualization_dir=str(out/'unused_visualization'))
        runtime=SimpleNamespace(**config)
        entry.get_config=lambda:runtime
        entry.construct_envs=lambda args:RPCEnvironment(args)
        entry.Graph=CalibratedGraph
        record={k:str(v) if isinstance(v,torch.device) else v for k,v in config.items()}
        record.update(panel_sha256=hashlib.sha256(panel.read_bytes()).hexdigest(),
            category_assisted=options.category_assisted,
            calibration_changes=['Graph width/height intrinsics follow actual native 480x640 pixels',
                'Only graph point-cloud branch converts normalized depth to meters; BEV retains native normalized input'],
            scope='Released full UniGoal framework with calibrated interfaces and common HSSD geometry; not official HM3D/paper scores',
            goal_category_fallback='Explicit input category' if options.category_assisted else 'Unknown -1 if native goal detector fails; no oracle fallback')
        (out/'configuration.json').write_text(json.dumps(record,indent=2))
        os.chdir(REPO)
        torch.manual_seed(1)
        np.random.seed(1)
        try:
            entry.main()
        except Exception as exc:
            (out/'runtime_failure.json').write_text(json.dumps({'status':'runtime_failure',
                'error':type(exc).__name__,'detail':str(exc),'performance_score_valid':False},indent=2))
            raise
        records=json.loads((out/'records.json').read_text())
        assert len(records)==len(episodes)
        print(json.dumps({'complete':True,'episodes':len(records),'successes':sum(r['success'] for r in records)}),flush=True)
    finally:
        transport.close()
        # Release Ollama's resident model before another queued GPU benchmark.
        import requests
        with contextlib.suppress(Exception):
            requests.post(options.base_url.rstrip('/').removesuffix('/v1')+'/api/generate',json={'model':'llama3.2-vision:11b','keep_alive':0},timeout=20)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',required=True)
    parser.add_argument('--limit',type=int,default=30)
    parser.add_argument('--max-steps',type=int,default=500)
    parser.add_argument('--category-assisted',action='store_true')
    parser.add_argument('--sim-gate',action='store_true')
    parser.add_argument('--base-url',default='http://127.0.0.1:11434/v1/')
    options=parser.parse_args()
    with (ROOT/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        execute(options)
