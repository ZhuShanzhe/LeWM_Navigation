"""Retrieve public official sources and model inventories with pinned revisions."""
import concurrent.futures as cf
import hashlib
import io
import json
from pathlib import Path
import time
import urllib.request
import zipfile

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')

def get(url, timeout=60):
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'navigation-research'})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.read()
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2 + attempt)

def source(repo, name):
    dst = ROOT / 'repos' / name
    manifest = ROOT / 'results' / (name + '_download.json')
    if manifest.exists():
        return
    commit = json.loads(get('https://api.github.com/repos/' + repo + '/commits/HEAD'))['sha']
    data = get('https://codeload.github.com/' + repo + '/zip/' + commit, 180)
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        for member in archive.infolist():
            parts = Path(member.filename).parts[1:]
            if not parts or member.is_dir():
                continue
            if '..' in parts or Path(*parts).is_absolute():
                raise ValueError('Unexpected archive path')
            if any(p in {'assets', '.github', '__pycache__', 'docs', 'examples', 'notebooks', 'demo', 'imgs', 'images'} for p in parts):
                continue
            target = dst.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise FileExistsError(str(target))
            target.write_bytes(archive.read(member))
    result = {'repo': repo, 'commit': commit, 'archive_sha256': hashlib.sha256(data).hexdigest()}
    manifest.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

def download(url, filename):
    path = ROOT / 'weights' / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temp = path.with_suffix(path.suffix + '.partial')
    req = urllib.request.Request(url, headers={'User-Agent': 'navigation-research'})
    with urllib.request.urlopen(req, timeout=180) as response, temp.open('wb') as stream:
        while True:
            chunk = response.read(4 * 1024 * 1024)
            if not chunk:
                break
            stream.write(chunk)
    temp.replace(path)
    result = {'filename': filename, 'url': url, 'bytes': path.stat().st_size,
              'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (ROOT / 'results' / (filename + '.download.json')).write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources-only', action='store_true')
    args = parser.parse_args()
    for repo, name in [('facebookresearch/detectron2', 'detectron2'), ('Ram81/goat-bench', 'goat-bench'),
                       ('IDEA-Research/GroundingDINO', 'GroundingDINO'),
                       ('facebookresearch/segment-anything', 'segment-anything'),
                       ('IDEA-Research/Grounded-Segment-Anything', 'Grounded-Segment-Anything')]:
        try:
            source(repo, name)
        except Exception as exc:
            print(json.dumps({'repo': repo, 'error_class': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}), flush=True)
    if args.sources_only:
        raise SystemExit(0)
    info = json.loads(get('https://huggingface.co/api/datasets/axel81/goat-bench'))
    revision = info['sha']
    files = [s['rfilename'] for s in info['siblings'] if s['rfilename'].startswith('checkpoints/')]
    (ROOT / 'results/goat_weight_inventory.json').write_text(json.dumps({'revision': revision, 'files': files}, indent=2))
    for file in files:
        if file.endswith(('.pth', '.pt', '.ckpt')):
            try:
                download('https://huggingface.co/datasets/axel81/goat-bench/resolve/' + revision + '/' + file,
                         'goat_' + file.replace('/', '__'))
            except Exception as exc:
                print(json.dumps({'file': file, 'error_class': type(exc).__name__}), flush=True)
