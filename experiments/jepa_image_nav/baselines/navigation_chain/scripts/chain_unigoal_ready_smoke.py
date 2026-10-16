"""Wait for the pinned runtime download and run gated UniGoal inference."""
import json,os,socket,subprocess,time
from instance_fetch_assets import ROOT
ready=ROOT/'results/ollama_compatible_runtime.json'
deadline=time.monotonic()+7200
while not ready.exists():
    if time.monotonic()>deadline:raise TimeoutError('Compatible runtime asset preparation exceeded two hours')
    time.sleep(10)
binary=ROOT/'runtime/ollama_0_24_0/bin/ollama'
with socket.socket() as check:
    if check.connect_ex(('127.0.0.1',11435))==0:
        raise RuntimeError('Project runtime port 11435 already occupied; verify ownership before reuse')
log=(ROOT/'logs/ollama_compatible_server.log').open('a')
process=subprocess.Popen([str(binary),'serve'],stdout=log,stderr=log,start_new_session=True,
    env=dict(os.environ,OLLAMA_HOST='127.0.0.1:11435',OLLAMA_MODELS=str(ROOT/'models/ollama'),
        OLLAMA_NUM_PARALLEL='1',OLLAMA_CONTEXT_LENGTH='4096'))
import requests
session=requests.Session();session.trust_env=False
for _ in range(60):
    if process.poll() is not None:raise RuntimeError('Compatible Ollama server exited')
    try:
        response=session.get('http://127.0.0.1:11435/api/version',timeout=2)
        if response.ok:break
    except requests.RequestException:pass
    time.sleep(1)
else:raise TimeoutError('Compatible local runtime did not become ready')
(ROOT/'results/ollama_compatible_server.json').write_text(json.dumps({'pid':process.pid,
    'address':'127.0.0.1:11435','context_length':4096,'parallel_requests':1},indent=2))
env=dict(os.environ,NO_PROXY='localhost,127.0.0.1',no_proxy='localhost,127.0.0.1',
    LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
python=str(ROOT/'policy_env/bin/python')
subprocess.run([python,'-u',str(ROOT/'tools/chain_unigoal_runtime_gate.py')],env=env,check=True)
subprocess.run([python,'-u',str(ROOT/'tools/chain_unigoal_run.py'),'--run','unigoal_full_chain_smoke1_v5',
    '--limit','1','--max-steps','20','--base-url','http://127.0.0.1:11435/v1/'],env=env,check=True)
print('UNIGOAL_FULL_CHAIN_SMOKE_COMPLETED',flush=True)
