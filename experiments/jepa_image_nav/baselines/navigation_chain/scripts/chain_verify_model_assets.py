"""Verify downloaded Qwen tensor shards against publisher LFS object digests."""
import hashlib
import json
from pathlib import Path
from instance_fetch_assets import ROOT,get
inventory=json.loads((ROOT/'results/qwen_assets.json').read_text())
repo=inventory['repository']
revision=inventory['revision']
info=json.loads(get('https://huggingface.co/api/models/'+repo+'/revision/'+revision+'?blobs=true'))
assert info['sha']==revision
publisher={v['rfilename']:v for v in info['siblings']}
records=[]
for f in inventory['files']:
    path=ROOT/'models/Qwen2.5-VL-7B-Instruct'/f['file']
    sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    expected=publisher[f['file']].get('lfs',{}).get('sha256')
    record={'file':f['file'],'sha256':sha,'bytes':path.stat().st_size,'publisher_sha256':expected,
        'passed':sha==expected if expected else sha==f['sha256']}
    records.append(record)
result={'repository':repo,'revision':revision,'all_files_passed':all(x['passed'] for x in records),
    'safetensor_shards_publisher_verified':all(x['publisher_sha256'] and x['passed'] for x in records if x['file'].endswith('.safetensors')),
    'files':records}
(ROOT/'results/qwen_integrity_gate.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
assert result['all_files_passed'] and result['safetensor_shards_publisher_verified']
