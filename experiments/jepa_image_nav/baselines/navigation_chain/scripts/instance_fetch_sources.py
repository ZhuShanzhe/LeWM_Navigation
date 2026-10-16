"""Download pinned upstream code, excluding bulky demos and generated outputs."""
import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
SPECS = {
    'UniGoal': ('bagh2178/UniGoal', ('src/', 'configs/', 'third_party/')),
    'IEVE': ('XiaohanLei/IEVE', ('agents/', 'algo/', 'configs/', 'envs/', 'utils/')),
    'jev-objnav': ('owlet914/jev-objnav', ('nav_jev_bridge/', 'tools/', 'tests/', 'docs/', 'examples/', 'jev_obj/')),
}

def fetch(name):
    repo, folders = SPECS[name]
    target = ROOT / 'repos' / name
    sha = subprocess.check_output(['git', '-C', str(target), 'rev-parse', 'HEAD'], text=True).strip()
    paths = subprocess.check_output(['git', '-C', str(target), 'ls-tree', '-r', '--name-only', 'HEAD'], text=True).splitlines()
    paths = [p for p in paths if (('/' not in p) or p.startswith(folders))
             and Path(p).suffix.lower() in {'.py', '.md', '.yaml', '.yml', '.txt', '.json', '.toml', '.sh', '.cpp', '.h', '.hpp', '.cc', '.cmake', '.xml', '.cfg', '.rviz', '.launch', '.msg', '.srv', '.in', '.gitignore'}
             and not any(s in p for s in ('build/', 'devel/', '__pycache__/', 'node_modules/'))]
    def one(path):
        dest = target / path
        if dest.exists():
            return {'path': path, 'status': 'existing', 'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()}
        url = f'https://raw.githubusercontent.com/{repo}/{sha}/{path}'
        for attempt in range(3):
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'research-source-audit'})
                with urllib.request.urlopen(req, timeout=35) as response:
                    data = response.read()
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + '.download')
                tmp.write_bytes(data)
                tmp.replace(dest)
                return {'path': path, 'status': 'downloaded', 'sha256': hashlib.sha256(data).hexdigest()}
            except Exception as exc:
                error = type(exc).__name__
                time.sleep(attempt + 1)
        return {'path': path, 'status': 'failed', 'error_class': error}
    with cf.ThreadPoolExecutor(max_workers=8) as pool:
        records = list(pool.map(one, paths))
    result = {'repository': repo, 'commit': sha, 'files': records,
              'complete': not any(r['status'] == 'failed' for r in records)}
    (ROOT / 'results' / f'{name}_source_inventory.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({'name': name, 'files': len(records), 'complete': result['complete']}), flush=True)

if __name__ == '__main__':
    for name in SPECS:
        fetch(name)
