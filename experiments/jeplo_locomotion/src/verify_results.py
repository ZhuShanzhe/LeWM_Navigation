"""Standard-library verification of compact records; never runs robot control."""
import ast
import collections
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    checks={}
    specs=[('results.jsonl','summary.json','main_manifest.json',230),
           ('extended_results.jsonl','extended_summary.json','extended_manifest.json',135),
           ('ablation_results.jsonl','ablation_summary.json','ablation_manifest.json',20)]
    for filename,summary_name,manifest_name,n in specs:
        rows=[json.loads(s) for s in (ROOT/'results'/filename).read_text().splitlines()]
        summaries=json.loads((ROOT/'results'/summary_name).read_text())
        manifest=json.loads((ROOT/'configs'/manifest_name).read_text())
        assert len(rows)==len(manifest)==n,(filename,'count')
        assert len({r['episode_id'] for r in rows})==n,(filename,'duplicate')
        assert {r['episode_id'] for r in rows}=={r['episode_id'] for r in manifest}
        groups=collections.defaultdict(list)
        for r in rows:
            assert r['policy_sha256']=='ffe8318312ac2e27a159115a70e3f12ddbb7444d7aa1325a22cce57223a05520'
            assert r['protocol']=='deployment_interface_v2'
            groups[(r['action_mode'],r['group'])].append(r)
        assert len(groups)==len(summaries)
        for s in summaries:
            rs=groups[(s['mode'],s['group'])]
            assert s['n']==len(rs) and s['completed']==sum(r['success'] for r in rs)
            assert s['rate']==s['completed']/s['n']
            assert s['reasons']==dict(collections.Counter(r['reason'] for r in rs))
            for key in ['tracking_xy_rmse','tracking_yaw_rmse','policy_ms_p95','estimator_ms_p95','max_tau_proxy_Nm','stop_final_speed_m_s','stop_displacement_m','scan_coverage']:
                values=[r[key] for r in rs if r[key] is not None]
                if not values:assert s[key] is None
                else:
                    assert all(math.isfinite(v) for v in values)
                    assert math.isclose(s[key],sum(values)/len(values),rel_tol=1e-10,abs_tol=1e-10),(filename,key)
        checks[filename]={'episodes':n,'groups':len(groups),'summary_matches':True}
    audit=json.loads((ROOT/'results/interface_audit.json').read_text())
    assert all(m['official_deployment_tolerance_pass'] for m in audit['models'].values())
    assert all(not m['strict_allclose_pass'] for m in audit['models'].values())
    contract=json.loads((ROOT/'results/contract_tests.json').read_text())
    assert contract['tests']==10 and contract['pass'] and contract['failures']==contract['errors']==0
    replay=json.loads((ROOT/'results/replay_checks.json').read_text())
    assert len(replay)==3 and all(r['position_difference_max']==0 and r['original_reason']==r['replay_reason'] for r in replay)
    validation=json.loads((ROOT/'results/deployment_validation.json').read_text())
    assert validation['total_episodes']==385 and validation['physical_robot_used'] is False
    for p in (ROOT/'src').glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    print(json.dumps({'passed':True,'checks':checks,'physical_robot_used':False,'official_benchmark_reproduced':False},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
