"""Isolated inference microbenchmark on frozen-panel RGB, not navigation SR."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
DATA=Path('/root/autodl-tmp/jepa_vln_data_20261014')

def prepare():
    from PIL import Image
    sys.path.insert(0,str(DATA/'tools'))
    from nav_build import opensim,setpose
    folder=BASE/'results/latency_inputs';folder.mkdir(exist_ok=True)
    rows=[]
    for name in ['gibson_place_nav_v1_val_place_20perscene','hssd_image_nav_v1_test_place_20perscene','hssd_image_nav_v1_test_instance_20perscene']:
        panel=BASE/'plans'/(name+'.jsonl');eps=[json.loads(l) for l in panel.read_text().splitlines()]
        selected=[];seen=set()
        for e in eps:
            if e['scene_id'] in seen:continue
            selected.append(e);seen.add(e['scene_id'])
            if len(selected)==6:break
        for e in selected:
            key=hashlib.sha256(e['episode_id'].encode()).hexdigest()[:16]
            rp=folder/(key+'.rgb.npy');gp=folder/(key+'.goal.npy')
            if not rp.exists():
                hssd=e['dataset'].startswith('hssd')
                scene=e['scene_id'] if hssd else str(DATA/'raw/scene_datasets/gibson'/(e['scene_id']+'.glb'))
                sim=opensim(scene,DATA/'raw/hssd-hab-compat/hssd-hab.scene_dataset_config.json' if hssd else None)
                try:np.save(rp,setpose(sim,e['start_position'],e['start_yaw'])['rgb'][:,:,:3])
                finally:sim.close()
                np.save(gp,np.asarray(Image.open(DATA/'datasets'/e['dataset']/e['goal_image']).convert('RGB')))
            rows.append({'episode_id':e['episode_id'],'task':e['task'],'dataset':e['dataset'],'rgb':str(rp),'goal':str(gp),
                         'rgb_sha256':hashlib.sha256(rp.read_bytes()).hexdigest(),'goal_sha256':hashlib.sha256(gp.read_bytes()).hexdigest()})
    (folder/'manifest.json').write_text(json.dumps(rows,indent=2));print('LATENCY_INPUTS_READY',len(rows),flush=True)

def measure():
    # No other CUDA compute process may share this measurement.
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip()
    if active:raise RuntimeError('GPU is not exclusive; active compute PIDs: '+active)
    rows=json.loads((BASE/'results/latency_inputs/manifest.json').read_text());output=[]
    for model in ['fgprompt','debit']:
        with (BASE/'logs'/('isolated_latency_'+model+'.log')).open('w') as log:
            proc=subprocess.Popen([str(BASE/'policy_env/bin/python'),str(BASE/'tools/baseline_policy.py'),'--model',model],
                                  stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,text=True,bufsize=1)
            def rpc(obj):
                proc.stdin.write(json.dumps(obj)+'\n');proc.stdin.flush();line=proc.stdout.readline()
                if not line:raise RuntimeError('Policy exited; check isolated latency log')
                r=json.loads(line)
                if 'error' in r:raise RuntimeError(r['error'])
                return r
            ready=json.loads(proc.stdout.readline());assert ready['ready']
            net=[];wall=[];groups={}
            try:
                rpc({'op':'reset','goal':rows[0]['goal']})
                for _ in range(30):rpc({'op':'act','rgb':rows[0]['rgb']})
                for row in rows:
                    rpc({'op':'reset','goal':row['goal']});gn=[];gw=[]
                    for _ in range(10):
                        start=time.perf_counter();r=rpc({'op':'act','rgb':row['rgb']});gw.append((time.perf_counter()-start)*1000);gn.append(r['latency_ms'])
                    net.extend(gn);wall.extend(gw)
                    groups.setdefault(row['dataset']+'/'+row['task'],[]).extend(gn)
            finally:
                if proc.poll() is None:
                    # The policy protocol exits on close without a response line.
                    proc.stdin.write(json.dumps({'op':'close'})+'\n');proc.stdin.flush();proc.wait(timeout=20)
                if proc.returncode:raise RuntimeError('Policy exited nonzero; check isolated latency log')
            def stats(a):return {'n':len(a),'mean_ms':float(np.mean(a)),'median_ms':float(np.median(a)),'p95_ms':float(np.percentile(a,95))}
            output.append({'model':model,'policy':ready,'network_forward':stats(net),'rpc_wall':stats(wall),'by_track_network':{k:stats(v) for k,v in groups.items()}})
    result={'status':'complete','exclusive_gpu':True,'driver':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
            'samples':len(rows),'warmup_per_model':30,'repeats_per_sample':10,'runs':output,
            'scope':'Static frozen-panel start RGB replay; not a closed-loop real-time control or equal-time navigation experiment. Network timer excludes image preprocessing; RPC includes array loading, preprocessing and transport. Scene rendering, disk snapshot creation and goal reset excluded.',
            'inputs':[{'episode_id':r['episode_id'],'rgb_sha256':r['rgb_sha256'],'goal_sha256':r['goal_sha256']} for r in rows]}
    (BASE/'results/isolated_latency.json').write_text(json.dumps(result,indent=2));print(json.dumps(output),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--measure',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    if a.measure:measure()
