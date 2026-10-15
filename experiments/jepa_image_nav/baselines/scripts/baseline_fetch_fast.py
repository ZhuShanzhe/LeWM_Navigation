"""Bounded, resumable downloads from publisher URLs; no credential collection."""
import concurrent.futures as cf
import hashlib
import json
import os
import subprocess
import urllib.request as U
from pathlib import Path

ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')

def request(url, headers=None):
    return U.urlopen(U.Request(url,headers=headers or {}),timeout=75)

def ranges(url, name, n, publisher_digest=None):
    dest=ROOT/'weights'/name
    parts=ROOT/'weights'/(name+'.parts');parts.mkdir(exist_ok=True)
    block=1024*1024
    def part(start):
        end=min(start+block,n)-1;p=parts/str(start)
        if p.exists() and p.stat().st_size==end-start+1:return p
        for retry in range(4):
            try:
                with request(url,{'Range':f'bytes={start}-{end}'}) as r:
                    assert r.status==206 and r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
                    raw=r.read()
                assert len(raw)==end-start+1
                p.write_bytes(raw);return p
            except Exception:
                if retry==3:raise
    with cf.ThreadPoolExecutor(max_workers=20) as pool:
        paths=list(pool.map(part,range(0,n,block)))
    tmp=dest.with_suffix('.assembling')
    with tmp.open('wb') as f:
        for p in paths:f.write(p.read_bytes())
    sha=hashlib.file_digest(tmp.open('rb'),'sha256').hexdigest()
    if publisher_digest:assert publisher_digest.removeprefix('sha256:')==sha
    assert tmp.stat().st_size==n
    os.replace(tmp,dest)
    return {'file':name,'bytes':n,'sha256':sha,'publisher_digest':publisher_digest}

def debit():
    url='https://download.europe.naverlabs.com/navigation/debit/debit_small.pth'
    with request(url,{'Range':'bytes=0-0'}) as r:
        assert r.status==206;n=int(r.headers['Content-Range'].split('/')[-1])
        r.read()
    return ranges(url,'debit_small.verified.pth',n)

def fg():
    api='https://api.github.com/repos/XinyuSun/FGPrompt/releases/tags/v0.0.1'
    with request(api) as r:meta=json.load(r)
    asset=next(a for a in meta['assets'] if a['name']=='early-fusion-r9.pth')
    with request(asset['url']+'?download=1',{'Accept':'application/octet-stream'}) as r:url=r.url
    return ranges(url,'fgprompt-early-fusion-r9.verified.pth',asset['size'],asset.get('digest'))

def repos():
    # Explicit existing GitHub key; never read or transmit private key contents.
    env=dict(os.environ,GIT_SSH_COMMAND='ssh -i /root/.ssh/github_research -o BatchMode=yes -o ConnectTimeout=15')
    names={'debit':'naver/debit','croco':'naver/croco','LightGlue':'cvg/LightGlue',
           'IEVE':'XiaohanLei/IEVE','UniGoal':'bagh2178/UniGoal','PIG-Nav':'zpschang/PIG-Nav'}
    def one(item):
        name, repo=item;p=ROOT/'repos'/name
        with (ROOT/'logs'/f'ssh_fetch_{name}.log').open('w') as f:
            if not (p/'.git').exists():
                subprocess.run(['git','clone','--depth','1',f'git@github.com:{repo}.git',str(p)],env=env,stdout=f,stderr=f,timeout=150,check=True)
            elif not (p/'README.md').exists():
                subprocess.run(['git','-C',str(p),'fetch','--depth','1',f'git@github.com:{repo}.git','HEAD'],env=env,stdout=f,stderr=f,timeout=150,check=True)
                subprocess.run(['git','-C',str(p),'checkout','-b','baseline-official','FETCH_HEAD'],stdout=f,stderr=f,check=True)
        return name,subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    out={}
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        fs={pool.submit(one,x):x[0] for x in names.items()}
        for f in cf.as_completed(fs):
            try:k,v=f.result();out[k]=v
            except Exception as e:out[fs[f]]={'error':str(e)}
    (ROOT/'official_revisions.json').write_text(json.dumps(out,indent=2))
    return out

if __name__=='__main__':
    with cf.ThreadPoolExecutor(max_workers=3) as ex:
        futures={ex.submit(fn):name for name,fn in [('debit',debit),('fgprompt',fg),('repos',repos)]}
        for f in cf.as_completed(futures):
            name=futures[f]
            try:result={'passed':True,'result':f.result()}
            except Exception as e:result={'passed':False,'error':repr(e)}
            (ROOT/'results'/f'fetch_{name}.json').write_text(json.dumps(result,indent=2))
            print(name,result,flush=True)
