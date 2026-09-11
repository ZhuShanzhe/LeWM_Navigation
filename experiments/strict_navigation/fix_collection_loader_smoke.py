from pathlib import Path
p=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910/audit_collection_smoke.py')
s=p.read_text().replace("HDF5Dataset(summary['dataset'],","HDF5Dataset(str(Path(summary['dataset']).with_suffix('')),")
p.write_text(s)
