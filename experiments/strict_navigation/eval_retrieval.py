"""Independent wrapper so baseline evaluation remains unchanged."""
import runpy,os,json
from pathlib import Path
import stable_worldmodel as swm
from retrieval_prior import install
install(swm)
runpy.run_path('/root/autodl-tmp/lewm_research/strict_nav_20260910/eval_strict.py',run_name='__main__')
out=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910/runs')/os.environ['LEWM_RUN_TAG']
path=out/'case_metadata.json';meta=json.loads(path.read_text());retr=json.loads((out/'retrieval_metadata.json').read_text())
cap=meta['compute_cap_per_case'];mode=retr['mode']
if mode=='only':cap['candidate_model_steps']=0
if mode=='rerank':cap['iterations']=1
cap['retrieval_pair_distance_cap']=((meta['budget']+cap['receding']*5-1)//(cap['receding']*5))*retr['entries']
cap['retrieval_mode']=mode
meta['retrieval_budget_note']='Retrieval matching and extra image encoding are additional work; equal candidate expansion is not equal total FLOPs or latency.'
path.write_text(json.dumps(meta,indent=2))
