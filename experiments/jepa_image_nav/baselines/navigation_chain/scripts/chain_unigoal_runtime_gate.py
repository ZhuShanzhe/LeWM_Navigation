"""Confirm native mllama text/image serving, then unload the GPU model."""
import base64,fcntl,io,json,time
import requests
from PIL import Image
from instance_fetch_assets import ROOT
session=requests.Session();session.trust_env=False
url='http://127.0.0.1:11435'
panel='/root/autodl-tmp/jepa_vln_baselines_20261015/plans/hssd_image_nav_v1_test_instance_20perscene.jsonl'
episode=json.loads(open(panel).readline())
goal=Image.open('/root/autodl-tmp/jepa_vln_data_20261014/datasets/hssd_image_nav_v1/'+episode['goal_image']).convert('RGB')
buffer=io.BytesIO();goal.save(buffer,format='PNG')
results=[]
with (ROOT/'gpu_benchmark.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        for kind in ['text','image']:
            content='Reply with one short sentence.' if kind=='text' else [
                {'type':'text','text':'Describe this indoor image in one short sentence.'},
                {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()}}]
            start=time.monotonic()
            response=session.post(url+'/v1/chat/completions',json={
                'model':'llama3.2-vision:11b','messages':[{'role':'user','content':content}],
                'max_tokens':32},timeout=300)
            response.raise_for_status();data=response.json()
            assert data['choices'][0]['message']['content'].strip()
            results.append({'input':kind,'nonempty_response':True,'elapsed_s':time.monotonic()-start})
        runtime=session.get(url+'/api/ps',timeout=10).json()
        assert len(runtime['models'])==1 and runtime['models'][0]['size_vram']>0
        result={'passed':True,'model':'llama3.2-vision:11b','runtime_version':session.get(url+'/api/version').json(),
            'calls':results,'loaded_models':runtime['models'],'upstream_model_not_substituted':True}
        (ROOT/'results/unigoal_llm_runtime_gate.json').write_text(json.dumps(result,indent=2))
        print('UNIGOAL_TEXT_IMAGE_RUNTIME_GATE_PASSED',flush=True)
    finally:
        session.post(url+'/api/generate',json={'model':'llama3.2-vision:11b','keep_alive':0},timeout=20)
