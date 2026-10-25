"""Retrieve unchanged official episode packages; never bypass dataset licensing."""
import gzip
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

from paper_freeze import DATA, ROOT, frozen, sha


def get(url, limit=100*1024*1024):
    last = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                value = response.read(limit+1)
                assert len(value) <= limit, 'Unexpectedly large official episode download'
                return value
        except Exception as exc:
            last = exc
            time.sleep(min(2**attempt, 8))
    raise last


def gibson():
    tree = json.loads((DATA/'manifests/gibson_original_episode_tree.json').read_text())
    entries = [e for e in tree['tree'] if e['path'].startswith('val/') and e['path'].endswith('.json.gz')]
    assert len(entries) == 14
    report = {'upstream': 'facebookresearch/image-goal-nav-dataset', 'revision': tree['sha'],
        'unchanged_official_episodes': True, 'files': [], 'episodes': 0}
    for e in entries:
        p = ROOT/'official/gibson_imagegoal'/e['path']
        value = p.read_bytes() if p.exists() else get(
            f"https://raw.githubusercontent.com/{report['upstream']}/{tree['sha']}/{e['path']}")
        assert len(value) == e['size']
        assert hashlib.sha1(f'blob {len(value)}\0'.encode()+value).hexdigest() == e['sha']
        data = json.loads(gzip.decompress(value))
        episodes = data if isinstance(data, list) else data['episodes']
        assert isinstance(episodes, list) and episodes and all(isinstance(x, dict) for x in episodes)
        frozen(p, value)
        scene = Path(e['path']).name.removesuffix('.json.gz')
        assert (DATA/'raw/scene_datasets/gibson'/f'{scene}.glb').exists()
        report['files'].append({'scene': scene, 'path': str(p), 'sha256': sha(p),
            'git_blob_sha1': e['sha'], 'episodes': len(episodes),
            'source_schema': 'list' if isinstance(data, list) else 'Habitat episodes object'})
        report['episodes'] += len(episodes)
        print('OFFICIAL_GIBSON', scene, len(episodes), flush=True)
    frozen(ROOT/'manifests/official_gibson_imagegoal.json', report)
    print('OFFICIAL_GIBSON_COMPLETE', report['episodes'], flush=True)


def hssd():
    # Exact link supplied by the HSSD authors' README, not a third-party mirror.
    url = ('https://www.dropbox.com/scl/fi/n5m00eoydfedi0de1nh34/'
           'objectnav_hssd-hab_v0.2.5.zip?dl=1&rlkey=zvosfrxu99si8xkfmkd623b4s')
    archive = ROOT/'official/hssd_objectnav/objectnav_hssd-hab_v0.2.5.zip'
    used_url = url
    if archive.exists():
        value = archive.read_bytes()
    else:
        try:
            value = get(url)
        except Exception:
            # Same author-supplied file ID and access parameters, direct official content host.
            used_url = url.replace('www.dropbox.com', 'dl.dropboxusercontent.com')
            value = get(used_url)
    zip_data = zipfile.ZipFile(io.BytesIO(value))
    assert sum(i.file_size for i in zip_data.infolist()) < 1024**3
    selected = [i for i in zip_data.infolist() if '/val/' in '/'+i.filename and i.filename.endswith('.json.gz')]
    assert selected, 'Author archive lacks validation episode files'
    frozen(archive, value)
    report = {'source_url': url, 'retrieval_url': used_url, 'source_readme': 'https://github.com/3dlg-hcvc/hssd',
        'archive_sha256': sha(archive), 'upstream_checksum_available': False,
        'unchanged_official_episodes': True, 'files': []}
    for info in selected:
        path = Path(info.filename)
        assert not path.is_absolute() and '..' not in path.parts
        contents = zip_data.read(info)
        data = json.loads(gzip.decompress(contents))
        dest = ROOT/'official/hssd_objectnav'/path
        frozen(dest, contents)
        report['files'].append({'path': str(dest), 'sha256': sha(dest),
            'episodes': len(data.get('episodes', [])), 'goal_sets': len(data.get('goals_by_category', {}))})
    frozen(ROOT/'manifests/official_hssd_objectnav.json', report)
    print('OFFICIAL_HSSD_COMPLETE', len(selected), flush=True)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['gibson', 'hssd', 'all'], default='all')
    args = parser.parse_args()
    for name, function in [('gibson', gibson), ('hssd', hssd)]:
        if args.task in [name, 'all']:
            function()
