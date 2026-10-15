"""Cooperative scene resume; excludes pre-lock workers already in flight."""
import concurrent.futures as cf, json, os, subprocess, time
from pathlib import Path
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
PY='/root/autodl-tmp/jepa_vln_data_20261014/env/bin/python'

if __name__=='__main__':
    active=set()
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:args=path.read_bytes().decode().split('\0')
        except (OSError,UnicodeError):continue
        if not any(a.endswith('/baseline_eval.py') for a in args) or '--scene' not in args:continue
        active.add((args[args.index('--run')+1],args[args.index('--scene')+1]))
    jobs=[]
    for model,run,task,turn in [('debit','debit_native10_hssd_test_place','place',10),('debit','debit_native10_hssd_test_instance','instance',10),('lg_frontier','lg_frontier_hssd_test_instance','instance',30)]:
        panel=BASE/'plans'/('hssd_image_nav_v1_test_'+task+'_20perscene.jsonl')
        scenes=sorted({json.loads(l)['scene_id'] for l in panel.read_text().splitlines()})
        subset=[(model,run,str(panel),scene,turn) for scene in scenes if (run,scene) not in active and not(BASE/'results'/run/scene/'summary.json').exists()]
        jobs.append(subset)
    # Interleave the three panels rather than monopolizing the queue by one model.
    queue=[group[i] for i in range(max(map(len,jobs))) for group in jobs if i<len(group)]
    print(json.dumps({'excluded_inflight':sorted(active),'queued_scenes':len(queue),'workers':6}),flush=True)
    def child(job):
        model,run,panel,scene,turn=job
        cmd=[PY,str(BASE/'tools/baseline_eval.py'),'--model',model,'--run',run,'--panel',panel,'--scene',scene,'--max-steps','500','--turn-deg',str(turn)]
        with (BASE/'logs'/('cooperative_'+run+'_'+scene+'.log')).open('a') as log:
            r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=14400)
        record={'run':run,'scene':scene,'code':r.returncode};print(json.dumps(record),flush=True);return record
    with cf.ThreadPoolExecutor(max_workers=6) as pool:results=list(pool.map(child,queue))
    (BASE/'results/cooperative_execution.json').write_text(json.dumps(results,indent=2))
    if any(r['code'] for r in results):raise SystemExit(1)
