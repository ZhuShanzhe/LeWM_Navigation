"""Download pinned public artifacts with checksum verification and safe extraction."""
import os,json,time,urllib.request,concurrent.futures,hashlib,tarfile,shutil,traceback
from pathlib import Path
import zstandard as zstd
ROOT=Path('/root/autodl-tmp/lewm_research');R=ROOT/'round12h_20260909'
deadline=min(json.loads((R/'start.json').read_text())['deadline_unix']-7200,time.time()+7200)
def dl(url,path,sha=None):
 path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
 if path.exists(): return {'path':str(path),'bytes':path.stat().st_size,'existing':True}
 for attempt in range(5):
  try:
   offset=part.stat().st_size if part.exists() else 0
   req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
   with urllib.request.urlopen(req,timeout=60) as response:
    mode='ab' if offset and response.status==206 else 'wb'
    with part.open(mode) as out:
     last=time.time()
     while chunk:=response.read(8*1024*1024):
      if time.time()>deadline:raise TimeoutError('download budget exceeded')
      out.write(chunk)
      if time.time()-last>45:print('BYTES',path,part.stat().st_size,flush=True);last=time.time()
   with part.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   if sha and digest!=sha:part.rename(part.with_suffix('.bad'));raise ValueError('SHA mismatch')
   part.rename(path);print('DOWNLOADED',path,flush=True)
   return {'path':str(path),'bytes':path.stat().st_size,'sha256':digest,'url':url}
  except Exception as e:
   print('RETRY_DOWNLOAD',path,repr(e),flush=True)
   if time.time()>deadline:raise
   time.sleep(3*(attempt+1))
 raise RuntimeError(str(path))
def job(kind,env):
 try:
  manifest=R/f'assets_{kind}_{env}.json'
  while not manifest.exists():
   if time.time()>deadline:raise TimeoutError('metadata unavailable')
   time.sleep(5)
  m=json.loads(manifest.read_text());done=[]
  for f in m['files']:
   name=f['path']
   selected=name in ['config.json','weights.pt'] if kind=='models' else name.endswith(('.tar.zst','.h5','.h5.zst','.tar.gz'))
   if not selected:continue
   path=ROOT/f'data/hf_{env}'/name if kind=='models' else R/f'downloads/{env}'/name
   url='https://huggingface.co/'+('datasets/' if kind=='datasets' else '')+m['name']+'/resolve/'+m['revision']+'/'+name
   item=dl(url,path,f.get('lfs',{}).get('oid'));done.append(item)
   if kind=='datasets':
    dest=R/f'datasets/{env}';dest.mkdir(parents=True,exist_ok=True)
    if name.endswith('.tar.zst'):
     with path.open('rb') as inp,zstd.ZstdDecompressor().stream_reader(inp) as reader,tarfile.open(fileobj=reader,mode='r|') as tar:
      for member in tar:
       if member.size>shutil.disk_usage(dest).free-5*1024**3:raise RuntimeError('insufficient disk')
       tar.extract(member,path=dest,filter='data')
    elif name.endswith('.h5.zst'):
     outpath=dest/path.name[:-4]
     with path.open('rb') as inp,outpath.open('wb') as out:
      zstd.ZstdDecompressor().copy_stream(inp,out)
    elif name.endswith('.h5'):
     link=dest/path.name
     if not link.exists():link.symlink_to(path)
  (R/f'ready_{kind}_{env}.json').write_text(json.dumps({'env':env,'kind':kind,'files':done,'h5':[str(p) for p in (R/f'datasets/{env}').rglob('*.h5')] if kind=='datasets' else []},indent=2))
  print('READY',kind,env,flush=True)
 except Exception as e:
  (R/f'failure_download_{kind}_{env}.json').write_text(json.dumps({'error':repr(e),'traceback':traceback.format_exc()}))
  print('FAILED',kind,env,repr(e),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 tasks=[pool.submit(job,'models',e) for e in os.environ.get('LEWM_DOWNLOAD_ENVS','pusht,cube,reacher').split(',')]
 for t in tasks:t.result()
 tasks=[pool.submit(job,'datasets',e) for e in os.environ.get('LEWM_DOWNLOAD_ENVS','pusht,cube,reacher').split(',')]
 for t in tasks:t.result()
