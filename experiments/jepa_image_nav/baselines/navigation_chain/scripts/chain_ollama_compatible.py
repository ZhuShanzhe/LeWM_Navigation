"""Install the upstream-recommended mllama-compatible runtime in isolation."""
import hashlib,json,tarfile
import zstandard
from instance_fetch_assets import ROOT,get
from chain_download_runtime import stream_download
release=json.loads(get('https://api.github.com/repos/ollama/ollama/releases/tags/v0.24.0'))
asset=next(a for a in release['assets'] if a['name']=='ollama-linux-amd64.tar.zst')
archive=ROOT/'downloads'/('v0.24.0-'+asset['name'])
stream_download(asset['browser_download_url'],archive)
sha=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest()
assert archive.stat().st_size==asset['size']
assert asset['digest']=='sha256:'+sha
target=ROOT/'runtime/ollama_0_24_0'
target.mkdir(parents=True,exist_ok=True)
with archive.open('rb') as source,zstandard.ZstdDecompressor().stream_reader(source) as reader:
    with tarfile.open(fileobj=reader,mode='r|') as contents:
        contents.extractall(target,filter='data')
(ROOT/'results/ollama_compatible_runtime.json').write_text(json.dumps({
    'version':release['tag_name'],'sha256':sha,'publisher_digest_verified':True,
    'source_url':asset['browser_download_url'],'install_path':str(target),
    'compatibility_reference':'https://github.com/ollama/ollama/issues/16490',
    'host_service_changed':False,'model_changed':False},indent=2))
print('MLLAMA_COMPATIBLE_RUNTIME_INSTALLED',flush=True)
