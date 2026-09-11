import concurrent.futures, hashlib, json, time, urllib.request
from pathlib import Path
root=Path('/root/autodl-tmp/lewm_research')
jobs=[
 ('https://huggingface.co/quentinll/lewm-tworooms/resolve/77adaae0bc31deab21c93740d1f8bb947cd0bdec/config.json',root/'data/hf_tworoom/config.json',None),
 ('https://huggingface.co/quentinll/lewm-tworooms/resolve/77adaae0bc31deab21c93740d1f8bb947cd0bdec/weights.pt',root/'data/hf_tworoom/weights.pt','566f223624ea4bfb39dbfe6ae731198dd6ea73b7b8919fed6b1ecafca810f7dd'),
 ('https://huggingface.co/datasets/quentinll/lewm-tworooms/resolve/6903a2de048b13819d812da0b4dd661290bc01e4/tworoom.tar.zst',root/'data/tworoom.tar.zst',None)
]
def download(job):
 url,path,sha=job; path.parent.mkdir(parents=True,exist_ok=True)
 for attempt in range(4):
  try:
   tmp=path.with_suffix(path.suffix+'.part'); n=0; start=time.time(); report=start
   with urllib.request.urlopen(url,timeout=90) as response, tmp.open('wb') as out:
    print('START',path.name,'bytes',response.headers.get('Content-Length'),flush=True)
    while chunk:=response.read(4*1024*1024):
     out.write(chunk); n+=len(chunk)
     if time.time()-report>20:
      print('PROGRESS',path.name,round(n/1e6,1),'MB',flush=True); report=time.time()
   digest=hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()
   if sha and digest!=sha: raise ValueError('SHA256 mismatch')
   tmp.rename(path)
   print('DONE',path.name,n,digest,round(time.time()-start,1),'sec',flush=True)
   return dict(url=url,path=str(path),bytes=n,sha256=digest)
  except Exception as exc:
   print('RETRY',path.name,attempt,type(exc).__name__,str(exc),flush=True); time.sleep(2)
 raise RuntimeError('download failed '+str(path))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 results=list(pool.map(download,jobs))
(root/'data/download_manifest.json').write_text(json.dumps(results,indent=2))
