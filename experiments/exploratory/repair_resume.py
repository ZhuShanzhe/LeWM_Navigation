from pathlib import Path
import os,signal,json,time
R=Path('/root/autodl-tmp/lewm_research/round12h_20260909')
for pid,expected in [(38920,b'asset_recovery.py'),(37774,b'download_all.py')]:
 try:
  if expected not in Path(f'/proc/{pid}/cmdline').read_bytes():raise RuntimeError('PID changed')
  os.kill(pid,signal.SIGTERM)
 except FileNotFoundError:pass
time.sleep(2)
p=R/'download_all.py';s=p.read_text()
(R/'download_all.before_resume_fix.py').write_text(s)
s=s.replace('time.time()+7200','time.time()+14400')
a=s.index('def dl(');b=s.index('def job(',a)
s=s[:a]+'''def dl(url,path,sha=None,size=None):
 path.parent.mkdir(parents=True,exist_ok=True);part=path.with_suffix(path.suffix+'.part')
 if path.exists():
  if size and path.stat().st_size!=size:raise ValueError('Existing file size mismatch '+str(path))
  return {'path':str(path),'bytes':path.stat().st_size,'existing':True}
 for attempt in range(20):
  try:
   offset=part.stat().st_size if part.exists() else 0
   if not size or offset<size:
    req=urllib.request.Request(url,headers={'Range':f'bytes={offset}-','Accept-Encoding':'identity'} if offset else {'Accept-Encoding':'identity'})
    with urllib.request.urlopen(req,timeout=60) as response:
     cr=response.headers.get('Content-Range','')
     if response.status==206:
      if not cr.startswith(f'bytes {offset}-'):raise ValueError('Unexpected Content-Range '+cr)
      mode='ab' if offset else 'wb'
     elif response.status==200:
      if offset:
       raise ValueError('Server ignored requested Range; preserving partial file')
      mode='wb'
     else:raise ValueError('Unexpected status '+str(response.status))
     with part.open(mode) as out:
      last=time.time()
      while chunk:=response.read(8*1024*1024):
       if time.time()>deadline:raise TimeoutError('download budget exceeded')
       out.write(chunk)
       if time.time()-last>45:print('BYTES',path,part.stat().st_size,flush=True);last=time.time()
   actual=part.stat().st_size
   if size and actual<size:raise EOFError(f'Truncated response {actual}/{size}; resume without discarding')
   if size and actual>size:raise ValueError('Oversized partial file '+str(actual))
   with part.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
   if sha and digest!=sha:
    part.rename(part.with_name(part.name+'.bad_full_'+str(int(time.time()))))
    raise ValueError('Full-size SHA mismatch; quarantined')
   part.rename(path);print('DOWNLOADED',path,flush=True)
   return {'path':str(path),'bytes':actual,'sha256':digest,'url':url}
  except Exception as e:
   print('RETRY_DOWNLOAD',path,repr(e),flush=True)
   if time.time()>deadline:raise
   time.sleep(min(20,3*(attempt+1)))
 raise RuntimeError(str(path))
''' + s[b:]
s=s.replace("dl(url,path,f.get('lfs',{}).get('oid'))","dl(url,path,f.get('lfs',{}).get('oid'),f.get('size'))")
p.write_text(s)
bad=R/'downloads/reacher/reacher.tar.zst.bad';part=R/'downloads/reacher/reacher.tar.zst.part'
audit={'time':time.time(),'change':'Validate expected size and Content-Range; preserve truncated responses; full SHA before accepting.'}
if bad.exists() and part.exists() and 0<part.stat().st_size<bad.stat().st_size<23750614946:
 with bad.open('rb') as f,part.open('rb') as g:prefix_matches=f.read(1048576)==g.read(1048576)
 audit['prefix_matches']=prefix_matches
 if prefix_matches:
  archive=part.with_name(part.name+'.short_backup')
  if archive.exists():raise RuntimeError('Backup already exists')
  part.rename(archive);bad.rename(part);audit['recovered_partial_bytes']=part.stat().st_size
(R/'download_repair.json').write_text(json.dumps(audit,indent=2))
print(audit)
