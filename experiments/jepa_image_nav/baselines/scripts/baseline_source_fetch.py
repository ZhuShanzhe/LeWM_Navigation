"""Fetch small official source files pinned to Git blobs, avoiding bulky demos."""
import concurrent.futures as cf, hashlib, json, urllib.request as U
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')

def read(url):
    last=None
    for attempt in range(3):
        try:
            with U.urlopen(url,timeout=45) as r:return r.read()
        except Exception as e:last=e
    raise last

def fetch(repo,name):
    tree=json.loads(read(f'https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1'))
    sha=tree['sha'];dest=ROOT/'repos'/name;dest.mkdir(exist_ok=True)
    entries=[e for e in tree['tree'] if e['type']=='blob' and (e['path'].endswith(('.py','.yaml','.yml','.toml')) or e['path'] in ['README.md','LICENSE','LICENSE.md'])]
    def one(e):
        p=dest/e['path'];p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            raw=p.read_bytes()
            if hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['sha']:return
            raise RuntimeError(f'Refuse to overwrite different source: {p}')
        raw=read(f'https://raw.githubusercontent.com/{repo}/{sha}/{e["path"]}')
        assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==e['sha']
        p.write_bytes(raw)
    failures=[]
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        fs={pool.submit(one,e):e['path'] for e in entries}
        for f in cf.as_completed(fs):
            try:f.result()
            except Exception as e:failures.append({'path':fs[f],'error':str(e)})
    result={'repository':repo,'git_tree':sha,'files':len(entries),'failures':failures,'all_file_git_blobs_verified':not failures}
    (ROOT/'results'/f'source_{name}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    with cf.ThreadPoolExecutor(max_workers=2) as ex:
        fs=[ex.submit(fetch,*x) for x in [('naver/debit','debit'),('cvg/LightGlue','LightGlue')]]
        for f in fs:
            try:f.result()
            except Exception as e:print(repr(e),flush=True)
