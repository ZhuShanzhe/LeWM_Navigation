"""Wait for complete scene evidence, then measure and validate; no GPU resets."""
import datetime,json,os,subprocess,time
from pathlib import Path
BASE=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
PY='/root/miniconda3/bin/python';POLICY=str(BASE/'policy_env/bin/python')

if __name__=='__main__':
    expected={'debit_native10_hssd_test_place':15,'debit_native10_hssd_test_instance':15,'lg_frontier_hssd_test_instance':15}
    start=time.monotonic();last=None
    while True:
        counts={name:len(list((BASE/'results'/name).glob('*/summary.json'))) for name in expected}
        if counts!=last:
            print(json.dumps({'time':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'complete_scenes':counts}),flush=True);last=counts
        if all(counts[k]==n for k,n in expected.items()):break
        if time.monotonic()-start>24*3600:raise TimeoutError('Scene panels did not complete; inspect process logs, no failures imputed')
        time.sleep(30)
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip():time.sleep(5)
    commands=[[POLICY,str(BASE/'tools/baseline_latency.py'),'--measure'],[POLICY,str(BASE/'tools/baseline_aggregate.py')],
              [POLICY,str(BASE/'tools/baseline_final_acceptance.py')],[PY,str(BASE/'tools/baseline_capture_status.py')]]
    for cmd in commands:
        print('START '+Path(cmd[1]).name,flush=True);subprocess.run(cmd,check=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'),timeout=1800)
    print('PENDING_EVALUATION_AND_ACCEPTANCE_COMPLETE',flush=True)
