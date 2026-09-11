"""Audit recorded evaluation instances and training logs; never changes experiments."""
import csv,json,math,shutil,subprocess
from pathlib import Path
import h5py,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/root/autodl-tmp/lewm_research')
records=[]
with h5py.File(ROOT/'data/tworoom.h5','r') as f:
    offsets=f['ep_offset'][:]
    proprio=f['proprio'][:]
    for mpath in sorted((ROOT/'runs').glob('*/metrics.json')):
        inv=json.loads((mpath.parent/'invocation.json').read_text())
        m=json.loads(mpath.read_text())
        if not inv.get('episodes'): continue
        argv=inv['argv']
        offset=next((int(x.split('=',1)[1]) for x in argv if x.startswith('eval.goal_offset_steps=')),25)
        idx=offsets[np.array(inv['episodes'])]+np.array(inv['start_steps'])
        starts=proprio[idx]; goals=proprio[idx+offset]
        d=np.linalg.norm(starts-goals,axis=-1)
        across=(starts[:,0]-112)*(goals[:,0]-112)<0
        success=np.asarray(m['metrics']['episode_successes'],dtype=bool)
        n=len(success);k=int(success.sum());p=k/n; z=1.96
        denom=1+z*z/n; centre=(p+z*z/(2*n))/denom
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
        rows=[{'env_index':i,'episode':inv['episodes'][i],'start_step':inv['start_steps'][i],
               'start_xy':starts[i].tolist(),'goal_xy':goals[i].tolist(),'initial_distance':float(d[i]),
               'initially_within_success_radius':bool(d[i]<16),'opposite_sides_of_vertical_wall':bool(across[i]),
               'success':bool(success[i])} for i in range(n)]
        (mpath.parent/'episode_audit.json').write_text(json.dumps(rows,indent=2))
        records.append({'run':mpath.parent.name,'n':n,'successes':k,'success_rate':100*p,
          'wilson95':[100*(centre-half),100*(centre+half)],'elapsed_seconds':m['elapsed_seconds'],
          'goal_offset_steps':offset,'mean_initial_distance':float(d.mean()),
          'initially_within_radius_n':int((d<16).sum()),'opposite_wall_sides_n':int(across.sum()),
          'opposite_wall_sides_successes':int(success[across].sum()),
          'outside_initial_radius_n':int((d>=16).sum()),'outside_initial_radius_successes':int(success[d>=16].sum())})
(ROOT/'reports/evaluation_summary.json').write_text(json.dumps(records,indent=2))
if records:
    with (ROOT/'reports/evaluation_summary.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
metrics=list((ROOT/'spt_cache/runs').glob('*/**/metrics.csv'))
if metrics:
    path=max(metrics,key=lambda p:p.stat().st_mtime)
    shutil.copy2(path,ROOT/'runs/train_pilot500/metrics.csv')
    with path.open() as f: raw=list(csv.DictReader(f))
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for key in ['fit/loss','fit/pred_loss']:
        rows=[r for r in raw if r.get(key)]
        axes[0].plot([int(r['step']) for r in rows],[float(r[key]) for r in rows],label=key)
    for key in ['validate/loss_epoch','validate/pred_loss_epoch']:
        rows=[r for r in raw if r.get(key)]
        axes[1].plot([int(r['step']) for r in rows],[float(r[key]) for r in rows],marker='o',label=key)
    for ax in axes: ax.set_xlabel('Optimizer step'); ax.set_ylabel('Loss'); ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('Training: 500-step pilot');axes[1].set_title('Validation: 8 batches / epoch')
    fig.savefig(ROOT/'reports/training_loss.png',dpi=180)
    plt.close(fig)
    (ROOT/'runs/train_pilot500/logger_source.txt').write_text(str(path))
print(json.dumps(records,indent=2))
