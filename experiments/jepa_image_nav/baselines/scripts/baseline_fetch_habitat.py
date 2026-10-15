"""Download three small Habitat v0.2.3 inference dependencies, no installation."""
import urllib.request, concurrent.futures, hashlib, json
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
files=['rl/ddppo/policy/resnet.py','rl/ddppo/policy/resnet_policy.py']
def one(rel):
    url='https://raw.githubusercontent.com/facebookresearch/habitat-lab/v0.2.3/habitat-baselines/habitat_baselines/'+rel
    with urllib.request.urlopen(url,timeout=60) as r:raw=r.read()
    out=ROOT/'repos'/('hab_'+Path(rel).name);out.write_bytes(raw)
    return {'url':url,'sha256':hashlib.sha256(raw).hexdigest(),'file':str(out)}
if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:result=list(ex.map(one,files))
    (ROOT/'results/habitat_inference_sources.json').write_text(json.dumps(result,indent=2))
