"""CPU-only release checks: exact copied files, directory documentation, schemas, summaries, secrets."""
from pathlib import Path
import ast, hashlib, json, re

def main():
    root=Path(__file__).resolve().parents[1]
    errors=[]; count=0
    manifest=json.loads((root/'provenance/publication_manifest.json').read_text())
    for row in manifest['files']:
        p=root/row['repository_path']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
            errors.append('Changed archived file: '+row['repository_path'])
    patterns=[r'-----BEGIN (?:OPENSSH|RSA|EC|DSA)? ?PRIVATE KEY-----',
              r'\bgh[pousr]_[A-Za-z0-9]{30,}\b',r'\bgithub_pat_[A-Za-z0-9_]{30,}\b',
              r'\bhf_[A-Za-z0-9]{25,}\b',r'\bsk-[A-Za-z0-9_-]{25,}\b',
              r'https?://[^\s/@:]+:[^\s/@]+@',r'(?i)(?:password|passwd|api_key|secret_key)\s*[:=]\s*[\"\'][^\"\'\s]{6,}[\"\']']
    for p in root.rglob('*'):
        if '.git' in p.relative_to(root).parts:continue
        if p.is_symlink():errors.append('Symlink: '+str(p));continue
        if p.is_dir():
            if not (p/'README.md').is_file():errors.append('Missing README: '+str(p.relative_to(root)))
            continue
        count+=1
        if p.stat().st_size>10_000_000:errors.append('Large file: '+str(p.relative_to(root)))
        if p.suffix in {'.pt','.ckpt','.h5','.hdf5','.mp4','.log','.pem','.key'}:errors.append('Excluded type: '+str(p))
        if p.suffix in {'.json','.py','.md','.sh','.yaml','.yml','.txt','.csv'}:
            text=p.read_text(encoding='utf-8')
            if p.suffix=='.json':
                try:json.loads(text)
                except Exception as e:errors.append(f'JSON {p}: {e}')
            if p.suffix=='.py':
                try:ast.parse(text,filename=str(p))
                except Exception as e:errors.append(f'Python {p}: {e}')
            for pattern in patterns:
                if re.search(pattern,text):errors.append('Potential secret (value suppressed): '+str(p.relative_to(root)));break
    final=json.loads((root/'experiments/strict_navigation/final_confirmation_v1/results.json').read_text())
    if not final['passed'] or len(final['trace_audits'])!=238:errors.append('Final audit incomplete')
    means=final['three_seed_descriptive']['known_layout']
    if abs(means['temporal']['mean_sr']-98.88888888888889)>1e-8:errors.append('Summary mismatch')
    print(json.dumps({'passed':not errors,'files_checked':count,'originals_verified':len(manifest['files']),
                      'errors':errors},ensure_ascii=False,indent=2))
    raise SystemExit(bool(errors))

if __name__=='__main__':main()
