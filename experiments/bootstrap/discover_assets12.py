import json,time,urllib.request,concurrent.futures
from pathlib import Path
root=Path('/root/autodl-tmp/lewm_research/round12h_20260909');root.mkdir(exist_ok=True)
def get(url):
 for attempt in range(4):
  try:
   with urllib.request.urlopen(url,timeout=30) as r:return json.load(r)
  except Exception as e:
   print('RETRY',url,attempt,str(e),flush=True);time.sleep(2)
 raise RuntimeError(url)
def fetch(task):
 kind,env=task;name='quentinll/lewm-'+env;api='https://huggingface.co/api/'+kind+'/'+name
 try:
  meta=get(api);rev=meta['sha'];files=get(api+'/tree/'+rev+'?recursive=true')
  out={'kind':kind,'env':env,'name':name,'revision':rev,'files':files}
  (root/f'assets_{kind}_{env}.json').write_text(json.dumps(out,indent=2))
  print('FOUND',kind,env,[(x['path'],x.get('size')) for x in files],flush=True);return out
 except Exception as e:return {'kind':kind,'env':env,'error':str(e)}
tasks=[(k,e) for k in ['models','datasets'] for e in ['pusht','cube','reacher']]
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:result=list(ex.map(fetch,tasks))
(root/'asset_manifest.json').write_text(json.dumps(result,indent=2))
