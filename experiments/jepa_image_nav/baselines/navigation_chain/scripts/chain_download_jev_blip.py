"""Native LAVIS EVA/BLIP2 assets, generation-pinned and publisher CRC verified."""
import base64,concurrent.futures as cf,hashlib,json,time,urllib.request
from pathlib import Path
import google_crc32c
from instance_fetch_assets import ROOT
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
base='https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/'
records=[]
for name in ['eva_vit_g.pth','blip2_pretrained.pth']:
    with opener.open(urllib.request.Request(base+name,method='HEAD'),timeout=30) as response:
        headers=dict(response.headers)
    total=int(headers['Content-Length'])
    generation=headers['x-goog-generation']
    expected=next(x.split('=',1)[1] for x in headers['x-goog-hash'].split(',') if x.strip().startswith('crc32c='))
    url=base+name+'?generation='+generation
    folder=ROOT/'downloads'/(name+'-'+generation)
    folder.mkdir(parents=True,exist_ok=True)
    chunk=4*1024*1024
    count=(total+chunk-1)//chunk
    def one(i):
        a=i*chunk;b=min(total,a+chunk)-1
        target=folder/f'{i:04d}.part'
        if target.exists() and target.stat().st_size==b-a+1:
            return
        for attempt in range(4):
            try:
                req=urllib.request.Request(url,headers={'Range':f'bytes={a}-{b}'})
                with opener.open(req,timeout=90) as response:
                    assert response.status==206
                    assert response.headers['Content-Range']==f'bytes {a}-{b}/{total}'
                    data=response.read()
                assert len(data)==b-a+1
                target.write_bytes(data)
                if i%25==0:print(name,i,count,flush=True)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(attempt+1)
    with cf.ThreadPoolExecutor(max_workers=12) as pool:list(pool.map(one,range(count)))
    target=ROOT/'weights'/name
    sha=hashlib.sha256();crc=google_crc32c.Checksum()
    with target.with_suffix('.assembling').open('wb') as stream:
        for i in range(count):
            data=(folder/f'{i:04d}.part').read_bytes()
            stream.write(data);sha.update(data);crc.update(data)
    assert base64.b64encode(crc.digest()).decode()==expected
    target.with_suffix('.assembling').replace(target)
    cache=Path('/root/.cache/torch/hub/checkpoints')/name
    if not cache.exists():cache.symlink_to(target)
    else:assert hashlib.file_digest(cache.open('rb'),'sha256').hexdigest()==sha.hexdigest()
    records.append({'file':name,'url':url,'generation':generation,'bytes':total,
        'sha256':sha.hexdigest(),'publisher_crc32c':expected,'publisher_crc32c_verified':True})
    (ROOT/'results/jev_blip_assets.json').write_text(json.dumps(records,indent=2))
print('JEV_NATIVE_BLIP_ASSETS_READY',flush=True)
