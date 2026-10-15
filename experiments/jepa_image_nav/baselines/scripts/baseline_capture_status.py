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
    report={'captured_at':datetime.datetime.now().astimezone().isoformat(),'cuInit_code':code,'gpu_recovery_action':action,
            'gpu_runtime_usable':code==0,'existing_completed_runs':counts,
            'pending_primary':['DEBiT-S 10deg HSSD place 300 episodes','DEBiT-S 10deg HSSD instance transfer 300 episodes'],
            'pending_diagnostics':['remaining RGB-D matcher/frontier panel','yaw-only approach visibility audit','isolated latency measurement'],
            'failure_handling':'CUDA/EGL/initialization failures are execution errors, never navigation failures; incomplete runs excluded from primary comparisons.'}
    (ROOT/'results/runtime_status.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
