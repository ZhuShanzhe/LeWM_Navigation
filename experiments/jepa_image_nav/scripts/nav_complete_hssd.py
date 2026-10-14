"""Resume formal HSSD generation after all selected scene assets arrive."""
import json,time,subprocess
from pathlib import Path
root=Path('/root/autodl-tmp/jepa_vln_data_20261014');env=root/'env/bin/python'
for attempt in range(1440):
    manifest=root/'manifests/hssd_source.json'
    if manifest.exists():
        files=json.loads(manifest.read_text())['files']
        missing=sum(not (root/'raw/hssd-hab'/f['path']).is_file() for f in files)
        if attempt%5==0:print('HSSD_MISSING_FILES',missing,flush=True)
        if missing==0:break
    time.sleep(60)
else:raise SystemExit('HSSD download did not complete within 24h; inspect source failures')
subprocess.run([str(env),str(root/'tools/nav_build.py')],check=True)
subprocess.run([str(env),str(root/'tools/nav_audit.py'),str(root/'datasets/hssd_image_nav_v1')],check=True)
print('HSSD_BUILD_AND_METADATA_AUDIT_FINISHED',flush=True)
