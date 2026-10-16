"""Fetch the same upstream LightGlue release before model initialization."""
import hashlib,json,shutil
from pathlib import Path
from instance_fetch_assets import ROOT,get
from chain_download_runtime import stream_download
repo='cvg/LightGlue'
release=json.loads(get('https://api.github.com/repos/'+repo+'/releases/tags/v0.1_arxiv'))
asset=next(a for a in release['assets'] if a['name']=='disk_lightglue.pth')
url=asset['browser_download_url']
path=ROOT/'weights/disk_lightglue_v0-1_arxiv.pth'
stream_download(url,path)
sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
assert path.stat().st_size==asset['size']
if asset.get('digest'):
    assert asset['digest']=='sha256:'+sha
import torch
state=torch.load(path,map_location='cpu',weights_only=True)
assert isinstance(state,dict) and state
cache=Path('/root/.cache/torch/hub/checkpoints')/path.name
cache.parent.mkdir(parents=True,exist_ok=True)
shutil.copy2(path,cache)
(ROOT/'results/unigoal_matcher_assets.json').write_text(json.dumps({
    'source':url,'release_id':release['id'],'bytes':path.stat().st_size,'sha256':sha,
    'publisher_digest_available':bool(asset.get('digest')),'weights_only_load':True},indent=2))
print('NATIVE_LIGHTGLUE_CACHE_READY',flush=True)
