"""Execute only a frozen final-confirmation job after checking its immutable inputs."""
from pathlib import Path
import os,sys,json,hashlib,runpy,time
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'final_confirmation_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
q=read(P/'protocol.json');tag=os.environ['LEWM_RUN_TAG'];job=next(x for x in q['evaluations'] if x['tag']==tag)
assert read(R/'oracle_waypoint_t16_v1/results.json')['passed']
for f,h in q['source_sha256'].items():assert sha(f)==h,f
for f,h in job['frozen_inputs'].items():assert sha(f)==h,f
for k,v in job['env'].items():assert os.environ.get(k)==v,(k,os.environ.get(k),v)
for k in ['LEWM_BOUND_ACTIONS','LEWM_TEMPORAL_HEAD','LEWM_HEAD_COST','LEWM_RETRIEVAL_LIBRARY','LEWM_RETRIEVAL_MODE','LEWM_NOOP']:
 assert not os.environ.get(k) or k in job['env'],('Unexpected intervention',k)
out=R/'runs'/tag;out.mkdir(parents=True,exist_ok=True)
assert not (out/'metrics.json').exists(),'Final outcomes already exist; no overwrite'
(out/'final_frozen_inputs.json').write_text(json.dumps({'checked_unix':time.time(),'protocol_sha256':sha(P/'protocol.json'),'tag':tag,
 'inputs_sha256':job['frozen_inputs'],'source_sha256':q['source_sha256'],'no_training':True},indent=2))
sys.argv=job['eval_argv'][1:]
runpy.run_path(sys.argv[0],run_name='__main__')
