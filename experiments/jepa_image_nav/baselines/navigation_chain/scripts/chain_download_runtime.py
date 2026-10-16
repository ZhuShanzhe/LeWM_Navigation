"""Pinned, project-local runtime/model downloads; no host services are installed."""
import argparse
import concurrent.futures as cf
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

from instance_fetch_assets import ROOT, get

def stream_download(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    temporary = dest.with_name(dest.name+'.partial')
    for attempt in range(4):
        try:
            start = temporary.stat().st_size if temporary.exists() else 0
            headers = {'User-Agent':'navigation-research'}
            if start:
                headers['Range'] = 'bytes='+str(start)+'-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=120) as response:
                mode = 'ab' if start and response.status == 206 else 'wb'
                with temporary.open(mode) as output:
                    while True:
                        data = response.read(4*1024*1024)
                        if not data:
                            break
                        output.write(data)
            temporary.replace(dest)
            return
        except Exception:
            if attempt == 3:
                raise

def qwen():
    repository = 'Qwen/Qwen2.5-VL-7B-Instruct'
    info = json.loads(get('https://huggingface.co/api/models/'+repository))
    revision = info['sha']
    target = ROOT / 'models/Qwen2.5-VL-7B-Instruct'
    files = [s['rfilename'] for s in info['siblings'] if s['rfilename'].endswith(('.safetensors','.json','.txt','.jinja'))]
    def one(name):
        path = target / name
        stream_download('https://huggingface.co/'+repository+'/resolve/'+revision+'/'+name, path)
        with path.open('rb') as data:
            sha = hashlib.file_digest(data, 'sha256').hexdigest()
        print(json.dumps({'qwen_file':name,'bytes':path.stat().st_size}),flush=True)
        return {'file':name,'bytes':path.stat().st_size,'sha256':sha}
    with cf.ThreadPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(one,files))
    (ROOT / 'results/qwen_assets.json').write_text(json.dumps({'repository':repository,'revision':revision,'files':records},indent=2))

def ollama():
    import zstandard
    release = json.loads(get('https://api.github.com/repos/ollama/ollama/releases/tags/v0.40.5'))
    asset = next(a for a in release['assets'] if a['name']=='ollama-linux-amd64.tar.zst')
    archive = ROOT / 'downloads' / (release['tag_name']+'-'+asset['name'])
    stream_download(asset['browser_download_url'], archive)
    with archive.open('rb') as data:
        sha = hashlib.file_digest(data,'sha256').hexdigest()
    expected = asset.get('digest')
    if expected:
        assert expected == 'sha256:'+sha
    target = ROOT / 'runtime/ollama'
    target.mkdir(parents=True,exist_ok=True)
    with archive.open('rb') as data, zstandard.ZstdDecompressor().stream_reader(data) as reader:
        with tarfile.open(fileobj=reader, mode='r|') as contents:
            contents.extractall(target,filter='data')
    (ROOT / 'results/ollama_runtime.json').write_text(json.dumps({'version':release['tag_name'],
        'url':asset['browser_download_url'],'sha256':sha,'publisher_digest':expected,
        'install_path':str(target),'host_service_installed':False},indent=2))
    print(json.dumps({'ollama_ready':True,'version':release['tag_name']}),flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target',choices=['qwen','ollama'])
    args = parser.parse_args()
    {'qwen':qwen,'ollama':ollama}[args.target]()
