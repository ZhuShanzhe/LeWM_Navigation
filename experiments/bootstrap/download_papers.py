import concurrent.futures,json,urllib.request,hashlib,time
from pathlib import Path
root=Path('/root/autodl-tmp/lewm_research/papers')
papers=json.loads((root/'sources.json').read_text())
def get(p):
 try:
  data=urllib.request.urlopen(p['url'],timeout=60).read()
  (root/(p['key']+'.html')).write_bytes(data)
  return dict(key=p['key'],url=p['url'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),downloaded_at=time.strftime('%Y-%m-%dT%H:%M:%S'))
 except Exception as exc: return dict(key=p['key'],error=str(exc))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 results=list(pool.map(get,papers))
(root/'download_manifest.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
