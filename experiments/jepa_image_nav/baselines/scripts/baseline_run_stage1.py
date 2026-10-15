"""Run the frozen primary panel with a common 500-action budget, no fitting."""
import json, os, subprocess, time
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
PY='/root/autodl-tmp/jepa_vln_data_20261014/env/bin/python'
jobs=[('gibson_place_nav_v1_val_place_20perscene','gibson_val_place'),
      ('hssd_image_nav_v1_test_place_20perscene','hssd_test_place'),
      ('hssd_image_nav_v1_test_instance_20perscene','hssd_test_instance')]
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='fgprompt');parser.add_argument('--prefix');parser.add_argument('--turn-deg',type=float,default=30.);parser.add_argument('--workers',type=int,default=2);args=parser.parse_args()
    import ctypes
    status=ctypes.CDLL('libcuda.so.1').cuInit(0)
    if status:
        (ROOT/'results/preflight_status.json').write_text(json.dumps({'cuda_init_code':status,'status':'blocked_before_evaluation','navigation_failures_added':0},indent=2))
        raise SystemExit('CUDA preflight failed; no evaluation started. Driver code: '+str(status))
    runs=[]
    for panel,suffix in jobs:
        run=(args.prefix or args.model)+'_'+suffix
        command=[PY,str(ROOT/'tools/baseline_eval.py'),'--model',args.model,'--panel',str(ROOT/'plans'/(panel+'.jsonl')),
                 '--run',run,'--workers',str(args.workers),'--max-steps','500','--turn-deg',str(args.turn_deg)]
        started=time.time()
        with (ROOT/'logs'/(run+'.log')).open('w') as f:
            p=subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,MAGNUM_LOG='quiet',HABITAT_SIM_LOG='quiet'),timeout=14400)
        runs.append({'run':run,'process_code':p.returncode,'seconds':time.time()-started})
        (ROOT/'results'/((args.prefix or args.model)+'_stage1_execution.json')).write_text(json.dumps(runs,indent=2))
        print(json.dumps(runs[-1]),flush=True)
