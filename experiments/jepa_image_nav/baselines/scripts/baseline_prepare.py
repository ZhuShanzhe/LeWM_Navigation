"""Prepare isolated inference dependencies and pinned official repositories."""
import concurrent.futures as cf
import json
import subprocess
import time
from pathlib import Path

ROOT = Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
REPOS = {
    'debit': 'https://github.com/naver/debit',
    'FGPrompt': 'https://github.com/XinyuSun/FGPrompt',
    'croco': 'https://github.com/naver/croco',
    'IEVE': 'https://github.com/XiaohanLei/IEVE',
    'UniGoal': 'https://github.com/bagh2178/UniGoal',
    'visualnav-transformer': 'https://github.com/robodhruv/visualnav-transformer',
    'PIG-Nav': 'https://github.com/zpschang/PIG-Nav',
    'NavDP': 'https://github.com/InternRobotics/NavDP',
    'jev-objnav': 'https://github.com/owlet914/jev-objnav',
    'LightGlue': 'https://github.com/cvg/LightGlue',
}
WEIGHTS = {
    'fgprompt-early-fusion-r9.pth': 'https://github.com/XinyuSun/FGPrompt/releases/download/v0.0.1/early-fusion-r9.pth',
    'debit_small.pth': 'https://download.europe.naverlabs.com/navigation/debit/debit_small.pth',
}

def run(cmd, log, timeout=1800):
    with log.open('w') as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
    return p.returncode

def repository(item):
    name, url = item
    path = ROOT/'repos'/name
    try:
        if not (path/'README.md').exists():
            if not (path/'.git').exists():
                code = run(['git', 'clone', '--depth', '1', url, str(path)], ROOT/'logs'/f'clone_{name}.log', 180)
            else:
                # Finish a previously interrupted checkout, never reset an edited repository.
                code = run(['git', '-C', str(path), 'fetch', '--depth', '1', 'origin'], ROOT/'logs'/f'fetch_{name}.log', 180)
                if code == 0:
                    code = run(['git', '-C', str(path), 'checkout', 'HEAD'], ROOT/'logs'/f'checkout_{name}.log', 180)
        else:
            code = 0
        sha = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip() if code == 0 else None
        return name, {'url': url, 'commit': sha, 'code': code}
    except Exception as e:
        return name, {'url': url, 'error': str(e)}

def weight(item):
    name, url = item
    path = ROOT/'weights'/name
    code = run(['curl', '-fL', '--retry', '2', '--connect-timeout', '20', '--max-time', '1200', '-C', '-', '-o', str(path), url], ROOT/'logs'/f'download_{name}.log', 1250)
    return name, {'url': url, 'code': code, 'bytes': path.stat().st_size if path.exists() else 0}

def environment():
    env = ROOT/'policy_env'
    if not (env/'bin/python').exists():
        subprocess.run(['/root/miniconda3/bin/python', '-m', 'venv', '--system-site-packages', str(env)], check=True)
    return run([str(env/'bin/python'), '-m', 'pip', 'install', 'gym==0.26.2', 'yacs', 'einops', 'omegaconf', 'timm==0.6.13', 'kornia', 'opencv-python-headless', 'gdown', 'scikit-image', 'scipy'], ROOT/'logs/policy_env_install.log')

if __name__ == '__main__':
    for folder in ('repos', 'weights', 'logs', 'results', 'tools', 'plans'):
        (ROOT/folder).mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(max_workers=5) as ex:
        env_job = ex.submit(environment)
        repo_jobs = [ex.submit(repository, item) for item in REPOS.items()]
        weight_jobs = [ex.submit(weight, item) for item in WEIGHTS.items()]
        repos = dict(job.result() for job in repo_jobs)
        (ROOT/'repos.json').write_text(json.dumps(repos, indent=2))
        weights = dict(job.result() for job in weight_jobs)
        result = {'created_at': time.time(), 'repos': repos, 'weights': weights, 'environment_install_code': env_job.result()}
        (ROOT/'preparation.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
