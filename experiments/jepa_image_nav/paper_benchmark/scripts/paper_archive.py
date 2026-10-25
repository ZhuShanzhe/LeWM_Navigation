"""Preserve explicitly named incomplete artifacts inside the paper benchmark only."""
from pathlib import Path
import time
from paper_freeze import ROOT


def archive(folder, names):
    base = (ROOT/'results').resolve(); folder = Path(folder).resolve()
    assert folder == base or base in folder.parents
    assert not (folder/'accepted_summary.json').exists(), 'Never archive accepted results'
    files = []
    for name in names:
        assert Path(name).name == name and name not in {'.', '..', 'accepted_summary.json', 'record.json'}
        path = folder/name
        if path.exists():
            assert path.is_file() and not path.is_symlink()
            files.append(path)
    if not files:
        return
    dest = folder/'attempts'/str(time.time_ns()); dest.mkdir(parents=True, exist_ok=False)
    assert folder in dest.resolve().parents
    for path in files:
        path.rename(dest/path.name)
