"""Native GroundingDINO BERT dependency, pinned data-only Hugging Face assets."""
import hashlib
import json
from instance_fetch_assets import ROOT,get
from chain_download_runtime import stream_download
repo='google-bert/bert-base-uncased'
info=json.loads(get('https://huggingface.co/api/models/'+repo+'?blobs=true'))
revision=info['sha']
files=['config.json','vocab.txt','tokenizer.json','tokenizer_config.json','model.safetensors']
inventory={f['rfilename']:f for f in info['siblings']}
records=[]
for name in files:
    path=ROOT/'models/bert-base-uncased'/name
    stream_download('https://huggingface.co/'+repo+'/resolve/'+revision+'/'+name,path)
    sha=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    lfs=inventory[name].get('lfs',{})
    if lfs:
        assert lfs['sha256']==sha and lfs['size']==path.stat().st_size
    records.append({'file':name,'bytes':path.stat().st_size,'sha256':sha,'publisher_lfs_verified':bool(lfs)})
    print(json.dumps(records[-1]),flush=True)
(ROOT/'results/bert_assets.json').write_text(json.dumps({'repository':repo,'revision':revision,'files':records},indent=2))
