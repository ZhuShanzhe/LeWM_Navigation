"""Restartable, isolated native UniGoal cold-start evaluations on the frozen panel.

Per-episode native processes preserve the original policy and stop conditions,
allow CUDA memory cleanup and retain each independently accepted episode.
"""
import fcntl,hashlib,json,os,subprocess,time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
out=ROOT/'results/unigoal_native_instance_hssd300';out.mkdir(parents=True,exist_ok=True)
source=Path('/root/autodl-tmp/jepa_vln_baselines_20261015/plans/hssd_image_nav_v1_test_instance_20perscene.jsonl')
panel=source.read_bytes();episodes=[json.loads(line) for line in panel.decode().splitlines()]
assert len(episodes)==300
if (out/'panel.jsonl').exists():assert (out/'panel.jsonl').read_bytes()==panel
else:(out/'panel.jsonl').write_bytes(panel)
env=dict(os.environ,NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',
    LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
python=str(ROOT/'policy_env/bin/python');check=str(ROOT/'goat_env/bin/python')

def write(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)

with (out/'batch.lock').open('a') as batch:
    fcntl.flock(batch,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+10800
    gate=ROOT/'results/pytorch3d_native_gate.json'
    while not gate.exists():
        if time.monotonic()>deadline:raise TimeoutError('Native oriented-box operator gate unavailable')
        time.sleep(10)
    assert json.loads(gate.read_text())['passed']
    # Smoke uses a separate output and a smaller budget, never merged as a score.
    smoke='unigoal_full_chain_smoke1_v7'
    with (ROOT/'logs/unigoal_native_smoke_v7.log').open('a') as log:
        if not (ROOT/'results'/smoke/'accepted_summary.json').exists():
            subprocess.run([python,'-u',str(ROOT/'tools/chain_unigoal_run.py'),'--run',smoke,
                '--limit','1','--max-steps','20','--base-url','http://127.0.0.1:11435/v1/'],env=env,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([check,str(ROOT/'tools/chain_unigoal_acceptance.py'),'--run',smoke,'--required','1'],
            env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    records=[];checks=[]
    for index,episode in enumerate(episodes):
        run='unigoal_native_instance_hssd300/episodes/'+episode['episode_id']
        folder=ROOT/'results'/run;folder.mkdir(parents=True,exist_ok=True)
        query=folder/'input_episode.jsonl'
        frozen=(json.dumps(episode)+'\n').encode()
        if query.exists():assert query.read_bytes()==frozen
        else:query.write_bytes(frozen)
        accepted=folder/'accepted_summary.json'
        if not accepted.exists():
            # Old partial traces/logs remain on disk. Completed accepted episodes
            # are never rerun; a failed episode's native state is restarted cold.
            from chain_archive_attempt import archive
            archive(folder,['runtime_failure.json','configuration.json','records.json',
                episode['episode_id']+'.json',episode['episode_id']+'.steps.jsonl',
                'detector_weight_gate.json','cuda_operator_gate.json','native_run.log','simulator_stderr.log',
                'execution_adapter.json','local_model_calls.jsonl','model_schema_events.jsonl'])
            with (folder/'native_run.log').open('a') as log:
                subprocess.run([python,'-u',str(ROOT/'tools/chain_unigoal_run.py'),'--run',run,
                    '--panel',str(query),'--max-steps','500','--base-url','http://127.0.0.1:11435/v1/'],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
                subprocess.run([check,str(ROOT/'tools/chain_unigoal_acceptance.py'),'--run',run,'--required','1'],
                    env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        result=json.loads(accepted.read_text());assert result['acceptance']=='passed' and result['episodes']==1
        record=result['records'][0];assert record['episode_id']==episode['episode_id']
        records.append(record)
        checks.append({'episode_id':episode['episode_id'],'sha256':hashlib.sha256(accepted.read_bytes()).hexdigest()})
        trace=out/(episode['episode_id']+'.steps.jsonl')
        target=folder/trace.name
        if trace.exists():assert trace.resolve()==target.resolve()
        else:trace.symlink_to(target)
        if index==0:
            config=json.loads((folder/'configuration.json').read_text())
            config.update(panel_sha256=hashlib.sha256(panel).hexdigest(),num_eval_episodes=300,num_episodes=300,
                execution='Independent cold-start native processes, per-episode torch/numpy seed1; accepted episodes resumed unchanged')
            write(out/'configuration.json',config)
            for name in ['cuda_operator_gate.json','detector_weight_gate.json']:
                write(out/name,json.loads((folder/name).read_text()))
        write(out/'records.json',records);write(out/'per_episode_acceptance.json',checks)
        print(json.dumps({'complete':len(records),'required':300,'success':record['success'],'steps':record['steps']}),flush=True)
    subprocess.run([check,str(ROOT/'tools/chain_unigoal_acceptance.py'),'--run','unigoal_native_instance_hssd300',
        '--required','300'],env=env,check=True)
