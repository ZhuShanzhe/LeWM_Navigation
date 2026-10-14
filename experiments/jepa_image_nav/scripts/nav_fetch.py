"""Retrieve public, pinned HSSD rigid scenes; assets stay off GitHub."""
import os, json, time, argparse, hashlib, requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
for key in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy'):
    os.environ.pop(key, None)
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET']='1'
os.environ['HF_HUB_ETAG_TIMEOUT']='120'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT']='120'
from huggingface_hub import HfApi, hf_hub_download
ROOT=Path('/root/autodl-tmp/jepa_vln_data_20261014')
REV='4369cb9876214c7fbebcf552eb532380e4d287e4'
DEST=ROOT/'raw/hssd-hab'
def wanted(p):
    if p.startswith(('scenes/','stages/','metadata/','semantics/')): return True
    if p.startswith('objects/'):
        return p.endswith(('.object_config.json','.glb','.ply'))
    return p in ('hssd-hab.scene_dataset_config.json','README.md','.gitattributes') or 'split' in p.lower()
def get(p):
    dest=DEST/p
    if dest.is_file() and dest.stat().st_size>0: return p
    dest.parent.mkdir(parents=True,exist_ok=True)
    route='raw' if p.endswith(('.json','.yaml','.csv','.md')) or p=='.gitattributes' else 'resolve'
    url=f'https://hf-mirror.com/datasets/hssd/hssd-hab/{route}/{REV}/{p}'
    for attempt in range(12):
        try:
            with requests.get(url,stream=True,timeout=(30,120)) as r:
                if r.status_code==429:
                    time.sleep(max(60,int(r.headers.get('Retry-After','60'))));continue
                r.raise_for_status()
                tmp=dest.with_name(dest.name+'.part')
                sha=hashlib.sha256();count=0
                with tmp.open('wb') as f:
                    for chunk in r.iter_content(1024*1024):
                        f.write(chunk);sha.update(chunk);count+=len(chunk)
                expected=r.headers.get('Content-Length')
                if expected and int(expected)!=count: raise ValueError('Incomplete download')
                for h in r.history:
                    linked=h.headers.get('x-linked-etag','').strip('"')
                    if len(linked)==64 and sha.hexdigest()!=linked: raise ValueError('SHA256 mismatch')
                if p.endswith('.json'): json.loads(tmp.read_text())
                tmp.replace(dest)
            return p
        except Exception as e:
            if attempt==11: return {'path':p,'error':str(e)}
            time.sleep(min(120,2**attempt))
    return {'path':p,'error':'Rate limiting persisted'}
def main():
    DEST.mkdir(parents=True,exist_ok=True)
    info=json.loads((ROOT/'manifests/hssd_api.json').read_text())
    if info.get('private') or info.get('gated'): raise RuntimeError('Public ungated dataset required')
    if info['sha']!=REV: raise RuntimeError('Revision changed: refresh pinned source deliberately')
    selected=[f['rfilename'] for f in info['siblings'] if wanted(f['rfilename'])]
    selected.sort(key=lambda p:(p.startswith('objects/'),not p.endswith('.json'),p))
    manifest={'repo':'hssd/hssd-hab','revision':REV,'license':'CC-BY-NC-4.0','files':[{'path':p} for p in selected]}
    (ROOT/'manifests/hssd_source.json').write_text(json.dumps(manifest,indent=2))
    print('FILES',len(selected),flush=True)
    failures=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures=[pool.submit(get,p) for p in selected]
        for i,future in enumerate(as_completed(futures),1):
            result=future.result()
            if isinstance(result,dict): failures.append(result)
            if i%100==0: print('PROGRESS',i,'/',len(selected),'FAILED',len(failures),flush=True)
    (ROOT/'manifests/hssd_download_failures.json').write_text(json.dumps(failures,indent=2))
    print('HSSD_DOWNLOAD_FINISHED',len(selected)-len(failures),'FAILED',len(failures),flush=True)
    if failures: raise SystemExit(1)
if __name__=='__main__': main()
