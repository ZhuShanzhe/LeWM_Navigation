"""Capture runtime health and completed/pending panels without changing GPU state."""
import ctypes, datetime, json, os, subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_vln_baselines_20261015')

if __name__=='__main__':
    raw=subprocess.check_output(['nvidia-smi','-q'],text=True)
    action=next((l.split(':',1)[1].strip() for l in raw.splitlines() if 'GPU Recovery Action' in l),'unknown')
    code=ctypes.CDLL('libcuda.so.1').cuInit(0)
    counts=[]
    for folder in sorted((ROOT/'results').iterdir()):
        summaries=[json.loads(p.read_text()) for p in folder.glob('*/summary.json')] if folder.is_dir() else []
        if summaries:counts.append({'run':folder.name,'scenes':len(summaries),'episodes':sum(s['episodes'] for s in summaries)})
    done={r['run']:r['episodes'] for r in counts}
    primary={'debit_native10_hssd_test_place':300,'debit_native10_hssd_test_instance':300}
    pending_primary=[name for name,n in primary.items() if done.get(name)!=n]
    pending_diagnostics=[]
    if done.get('lg_frontier_hssd_test_instance')!=300:pending_diagnostics.append('remaining RGB-D matcher/frontier panel')
    for file,label in [('horizontal_visibility_audit.json','yaw-only approach visibility audit'),('isolated_latency.json','isolated latency measurement'),('camera_extrinsics_audit.json','goal camera extrinsics sidecar audit'),('final_acceptance.json','evidence integrity acceptance')]:
        if not(ROOT/'results'/file).exists():pending_diagnostics.append(label)
    report={'captured_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'cuInit_code':code,'gpu_recovery_action':action,
            'gpu_runtime_usable':code==0,'existing_completed_runs':counts,
            'driver_version':subprocess.check_output(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True).strip(),
            'pending_primary':pending_primary,'pending_diagnostics':pending_diagnostics,
            'scientific_extensions_not_tested':['complete specialized IIN method','equal-time-budget navigation'],
            'failure_handling':'CUDA/EGL/initialization failures are execution errors, never navigation failures; incomplete runs excluded from primary comparisons.'}
    (ROOT/'results/runtime_status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
