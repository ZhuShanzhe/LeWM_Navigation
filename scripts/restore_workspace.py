"""Restore byte-identical archived files; never overwrite a differing destination."""
from pathlib import Path
import argparse, hashlib, json, shutil

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--destination', required=True, type=Path)
    ap.add_argument('--apply', action='store_true', help='Copy after all conflicts are checked; default is dry-run.')
    args = ap.parse_args()
    repo = Path(__file__).resolve().parents[1]
    manifest = json.loads((repo/'provenance/publication_manifest.json').read_text())
    target = args.destination.resolve()
    pending, conflicts = [], []
    for row in manifest['files']:
        src = repo/row['repository_path']; dst = (target/row['workspace_path']).resolve()
        if not dst.is_relative_to(target):
            raise ValueError('Unsafe destination')
        digest = hashlib.sha256(src.read_bytes()).hexdigest()
        if digest != row['sha256']:
            raise ValueError(f"Source differs: {row['repository_path']}")
        if dst.exists():
            if not dst.is_file() or hashlib.sha256(dst.read_bytes()).hexdigest() != digest:
                conflicts.append(str(dst))
        else:
            pending.append((src,dst))
    if conflicts:
        raise SystemExit('Refusing to overwrite differing files:\n'+'\n'.join(conflicts))
    print(f'Validated {len(manifest["files"])} originals; {len(pending)} missing files to copy.')
    if args.apply:
        for src,dst in pending:
            dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
        print('Restored retained core subset only. Removed historical records, datasets, checkpoint weights, raw traces, caches and virtualenv were NOT restored.')
    else:
        print('Dry-run only. Pass --apply to copy. Historical scripts require the documented original absolute root.')

if __name__=='__main__':
    main()
