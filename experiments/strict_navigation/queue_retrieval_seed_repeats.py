"""Complete a finite three-encoder-seed key retrieval comparison on development goals."""
import json,copy,time,hashlib
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910');P=R/'retrieval_seed_repeat_v1'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
old=read(R/'jobs.json');lookup={j['tag']:j for j in old};jobs=[];PY=str(R.parent/'venv/bin/python')
for seed in [3073,3074]:
 jobs.append({'tag':f'build_retrieval{seed}_v1','argv':[PY,str(R/'build_retrieval_seed.py'),str(seed)],
 'env':{'CUDA_VISIBLE_DEVICES':''},'watchdog_seconds':600})
 for mode in ['only','rerank']:
  for b in range(2):
   ref=f'strict3072_validation_long_base_retrieval_{mode}_b{b}';j=copy.deepcopy(lookup[ref])
   j['tag']=ref.replace('3072',str(seed))
   j['argv']=[x.replace('strict_tw_s3072_v2',f'strict_tw_s{seed}').replace('/'+ref+'/', '/'+j['tag']+'/') for x in j['argv']]
   j['env']['LEWM_RETRIEVAL_LIBRARY']=str(R/'priors'/f'retrieval{seed}'/'library.pt')
   j['note']='Same training pairs/action chunks, independent encoder seed; development known-layout only vs WM rerank. Not equal total compute.'
   jobs.append(j)
assert not set(lookup).intersection(j['tag'] for j in jobs)
idx=next(i for i,j in enumerate(old) if j['tag']=='summarize_three_seed_baseline')
assert all(any(x['tag']==f'fit_temporal{s}' for x in old[:idx]) for s in [3073,3074])
assert not any(j['tag']=='summarize_three_seed_baseline' for j in read(R/'status.json')['jobs'])
P.mkdir(exist_ok=False)
protocol={'created_unix':time.time(),'jobs':jobs,'frozen_source_sha256':{str(R/f):sha(R/f) for f in ['retrieval_prior.py','eval_retrieval.py','build_retrieval_seed.py']},
 'case_sha256':{str(R/'splits'/f'cases_validation_{b}.json'):sha(R/'splits'/f'cases_validation_{b}.json') for b in [0,1]},
 'prior_seed_reference':'strict3072_validation_long_base_retrieval_only / rerank',
 'scope':'Only versus dynamic reranking,100 common development long-goal cases per model; no new data, no tuning, no heldout reserve use.',
 'limits':'Three independent WM training seeds, but same pair and evaluation seeds. Additional retrieval/encoding costs; no assumption of equal total compute. Geometric cross-seed repetitions not included in this finite stage.'}
(P/'protocol.json').write_text(json.dumps(protocol,indent=2));(P/'jobs_before.json').write_text(json.dumps(old,indent=2))
new=old[:idx]+jobs+old[idx:];assert len(new)==len({j['tag'] for j in new})
tmp=R/'jobs.retrieval_repeats.tmp';tmp.write_text(json.dumps(new,indent=2));tmp.replace(R/'jobs.json')
print(json.dumps({'added':len(jobs),'before':old[idx]['tag'],'total':len(new)}))
