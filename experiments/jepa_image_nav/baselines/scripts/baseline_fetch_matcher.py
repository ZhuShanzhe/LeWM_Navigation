"""Download original SuperPoint and LightGlue public pretrained weights."""
import concurrent.futures, json, sys
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
sys.path.insert(0,str(ROOT))
from baseline_fetch_fast import request, ranges

if __name__=='__main__':
    with request('https://api.github.com/repos/cvg/LightGlue/releases/tags/v0.1_arxiv') as r:release=json.load(r)
    names=['superpoint_v1.pth','superpoint_lightglue.pth']
    def one(name):
        asset=next(a for a in release['assets'] if a['name']==name)
        with request(asset['url']+'?download=1',{'Accept':'application/octet-stream'}) as r:url=r.url
        return ranges(url,name,asset['size'],asset.get('digest'))
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:result=list(ex.map(one,names))
    (ROOT/'results/matcher_weights.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
