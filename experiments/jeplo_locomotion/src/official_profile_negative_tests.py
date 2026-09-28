import copy,json,sys
from pathlib import Path
from audit_official_profile import audit
repo=Path(sys.argv[1]);profile=json.loads(Path(sys.argv[2]).read_text());out=Path(sys.argv[3])
records=[]
mutations={
    'wrong_commit':lambda p:p.update(upstream_commit='0'*40),
    'wrong_checkpoint':lambda p:p['models_sha256'].update({'policy.onnx':'0'*64}),
    'wrong_point_stride':lambda p:p['sensor'].update(point_downsample=1),
    'wrong_action_mode':lambda p:p['controller'].update(raw_actions=False),
    'real_robot_network':lambda p:p['controller'].update(interface='eth0',dds_domain=0),
    'unjustified_paper_claim':lambda p:p.update(paper_comparison_allowed=True),
    'wrong_command':lambda p:p['commands'].update(controller=['deploy','--not-official']),
}
for name,mutate in mutations.items():
    changed=copy.deepcopy(profile);mutate(changed);r=audit(repo,changed)
    rejected=not r['source_model_gate_pass'];assert rejected,name
    records.append({'case':name,'rejected':rejected})
report={'tests':len(records),'pass':True,'cases':records,'scope':'source/model gate fail-closed checks; not runtime validation'}
out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
