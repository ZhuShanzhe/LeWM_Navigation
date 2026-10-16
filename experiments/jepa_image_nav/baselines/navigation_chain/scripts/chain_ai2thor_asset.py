"""Bounded parallel download of the native AI2THOR binary, publisher SHA verified.

Resource preparation has no GPU lock. It never substitutes another simulator
version and never treats a download failure as model performance.
"""
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import time
import zipfile
import requests
import ai2thor.build
import ai2thor.downloader
from ai2thor.platform import CloudRendering

ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
ai2thor.build.base_url='https://ai2-thor-public.s3.amazonaws.com/'
build=ai2thor.build.Build(CloudRendering,'f0825767cd50d69f666c7f282e54abfe58f1e917',False,
    releases_dir=str(ROOT/'runtime/ai2thor/releases'))
url=build.url
def request(method,url,**kwargs):
    for attempt in range(4):
        try:
            result=requests.request(method,url,timeout=(20,120),**kwargs)
            result.raise_for_status()
            return result
        except Exception:
            if attempt==3:
                raise
            time.sleep(2+attempt)
expected=request('GET',build.sha256_url).text.strip()
assert len(expected)==64 and all(c in '0123456789abcdef' for c in expected)
response=request('HEAD',url)
total=int(response.headers['Content-Length'])
chunks=ROOT/'downloads'/('ai2thor-'+build.commit_id)
chunks.mkdir(parents=True,exist_ok=True)
chunk_size=2*1024*1024
def one(index):
    start=index*chunk_size
    end=min(total,start+chunk_size)-1
    path=chunks/f'{index:04d}.part'
    if path.exists() and path.stat().st_size==end-start+1:
        return path
    response=request('GET',url+'?range_part='+str(index),headers={'Range':f'bytes={start}-{end}'})
    assert response.status_code==206 and response.headers['Content-Range']==f'bytes {start}-{end}/{total}'
    assert len(response.content)==end-start+1
    path.write_bytes(response.content)
    print(json.dumps({'part':index,'parts':(total+chunk_size-1)//chunk_size}),flush=True)
    return path
count=(total+chunk_size-1)//chunk_size
with cf.ThreadPoolExecutor(max_workers=16) as pool:
    list(pool.map(one,range(count)))
archive=ROOT/'downloads'/(build.name+'.zip')
sha=hashlib.sha256()
with archive.open('wb') as output:
    for index in range(count):
        data=(chunks/f'{index:04d}.part').read_bytes()
        output.write(data)
        sha.update(data)
assert sha.hexdigest()==expected
with zipfile.ZipFile(archive) as contents:
    assert contents.testzip() is None
    assert all(not Path(x.filename).is_absolute() and '..' not in Path(x.filename).parts for x in contents.infolist())
original=ai2thor.downloader.download
def cached(request_url,sha256_digest,include_private_scenes=False):
    assert request_url==url and not include_private_scenes
    assert sha256_digest.strip()==expected
    return archive.read_bytes()
ai2thor.downloader.download=cached
build.download()
result={'ready':True,'native_build':build.commit_id,'platform':'CloudRendering',
    'source_url':url,'bytes':total,'sha256':expected,'publisher_digest_verified':True,
    'install_path':build.base_dir,'transport':'HTTPS to the same official AWS bucket, 16 bounded range workers'}
(ROOT/'results/ai2thor_runtime.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
