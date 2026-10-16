"""Install a pinned project-local RoboStack solver, no host shell changes."""
import hashlib
import json
import tarfile
from instance_fetch_assets import ROOT,get
from chain_download_runtime import stream_download

release=json.loads(get('https://api.github.com/repos/prefix-dev/pixi/releases/latest'))
asset=next(a for a in release['assets'] if a['name']=='pixi-x86_64-unknown-linux-musl.tar.gz')
archive=ROOT/'downloads'/(release['tag_name']+'-'+asset['name'])
stream_download(asset['browser_download_url'],archive)
sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
expected=asset.get('digest')
assert expected=='sha256:'+sha,'Publisher digest missing or mismatch'
dest=ROOT/'runtime/pixi'
dest.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive) as data:
    assert all(not m.name.startswith('/') and '..' not in m.name.split('/') for m in data.getmembers())
    data.extractall(dest,filter='data')
(ROOT/'results/pixi_runtime.json').write_text(json.dumps({'version':release['tag_name'],
    'sha256':sha,'publisher_digest_verified':True,'source_url':asset['browser_download_url']},indent=2))
print(json.dumps({'ready':True,'version':release['tag_name']}),flush=True)
