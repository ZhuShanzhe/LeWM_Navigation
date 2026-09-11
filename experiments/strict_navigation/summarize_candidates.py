"""Summarize candidate interventions; no method fitting or outcome filtering."""
import json
from pathlib import Path
import numpy as np
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'diagnostics/strict3072/candidate_audit.json';d=json.loads(p.read_text());rows=d['rows']
out={}
for label,rs in [('all',rows),('same_side',[r for r in rows if not r['across_wall']]),('across_wall',[r for r in rows if r['across_wall']])]:
 if not rs:continue
 z={'n':len(rs)}
 for key in ['rank_pred_vs_actual_latent','rank_actual_latent_vs_distance','rank_actual_latent_vs_door_proxy','selection_regret_actual_latent','action_clipping_fraction']:
  x=[r[key] for r in rs if r[key] is not None];z[key]={'median':float(np.median(x)),'mean':float(np.mean(x))} if x else None
 for h in ['1','3','5','10','15']:
  z['error_'+h]={k:float(np.mean([r['logged_action_errors'][h][k] for r in rs])) for k in ['open_loop','teacher_forced']}
 for choice in ['predicted_latent','oracle_actual_latent','oracle_euclidean','oracle_door_proxy']:
  z[choice]={'mean_distance_after5':float(np.mean([r['prefix_interventions'][choice]['distance_after_5'] for r in rs])),
  'prefix_successes':sum(r['prefix_interventions'][choice]['success_within_5'] for r in rs)}
 out[label]=z
payload={'model':d['model'],'test_cases':len(rows),'summary':out,'limitations':d['limitations'],
'notice':'Mechanism exploration used the first 100 development-test cases. Final confirmatory cases are reserved separately. No full closed-loop oracle effect established.'}
(R/'reports/candidate_diagnostic_summary.json').write_text(json.dumps(payload,indent=2))
print(json.dumps(payload,indent=2))
