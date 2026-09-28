"""Read-only official-source/model gate; never launches robot or simulator.

Exit 0 means source and checkpoint identity only. It is not runtime acceptance.
"""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()

def audit(repo,profile):
    checks=[]
    def check(name,passed,detail):
        checks.append(dict(name=name,passed=bool(passed),detail=detail))
    result=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True)
    actual=result.stdout.strip()
    check('upstream_commit',result.returncode==0 and actual==profile['upstream_commit'],actual)
    result=subprocess.run(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'],capture_output=True,text=True)
    check('tracked_source_clean',result.returncode==0 and not result.stdout.strip(),result.stdout.strip())
    hashes={}
    for name,expected in profile['models_sha256'].items():
        path=repo/profile['model_subdir']/name
        observed=sha256(path) if path.is_file() else None
        hashes[name]=observed
        check('checkpoint:'+name,observed==expected,observed)
    readme_path=repo/'README.md'
    readme=readme_path.read_text() if readme_path.exists() else ''
    for name,args in profile['commands'].items():
        # The README includes executable path prefixes; verify the full argument sequence.
        text=' '.join(args)
        check('readme_command:'+name,text in readme,text)
    c=profile['controller'];s=profile['sensor']
    check('simulation_only',c['interface']=='lo' and c['dds_domain']==1,'loopback/domain 1')
    check('recommended_action_sensor_flags',c['raw_actions'] and c['ood_consecutive']==15 and s['point_downsample']==4 and s['cage_mask'] is True,'raw, OOD 15, stride 4, cage on')
    expected_sensor={'rows':25,'columns':60,'point_downsample':4,'stacked_scans':5,'lidar_hz':20,'depth_publish_hz':10,'cage_mask':True}
    check('complete_sensor_profile',all(s.get(k)==v for k,v in expected_sensor.items()),s)
    check('controller_frequencies',c.get('policy_hz')==50 and c.get('estimator_hz')==10,c)
    expected_runtime={'os':'Ubuntu 22.04','onnxruntime':'1.22.0','tensorrt':'10.9','cuda':'12.8','cudnn_major':9}
    check('declared_runtime_profile',all(profile['runtime_readme'].get(k)==v for k,v in expected_runtime.items()),'declared requirements only, NOT installed-library verification')
    check('no_paper_claim',profile['paper_comparison_allowed'] is False and profile['paper_protocol']['confirmed'] is False,'paper protocol unresolved')
    resources={}
    for rel in ['README.md','deployment/go2_deploy/deploy/config.json','deployment/go2_deploy/deploy/src/deploy.cpp','deployment/lidar_depth_pub/lidar_depth_pub.cpp','deployment/unitree_mujoco/simulate/src/mid360_lidar.h','deployment/unitree_mujoco/simulate/src/terrain_switcher.h','deployment/unitree_mujoco/simulate/config.yaml','deployment/unitree_mujoco/simulate/assets/mid360.npy','deployment/unitree_mujoco/unitree_robots/go2/scene.xml']:
        path=repo/rel
        resources[rel]=sha256(path) if path.is_file() else None
        check('resource:'+rel,path.is_file(),resources[rel])
    passed=all(x['passed'] for x in checks)
    return dict(profile=profile['profile_id'],source_model_gate_pass=passed,checks=checks,resources_sha256=resources,models_sha256=hashes,runtime_gate_pass=None,closed_loop_gate_pass=None,paper_protocol_gate_pass=False,paper_performance_reproduced=False)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',required=True,type=Path)
    p.add_argument('--profile',type=Path,default=Path(__file__).resolve().parents[1]/'configs/official_readme_profile.json')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    report=audit(a.repo,json.loads(a.profile.read_text(encoding='utf-8')))
    encoded=json.dumps(report,ensure_ascii=False,indent=2)
    if a.output:a.output.write_text(encoded,encoding='utf-8')
    print(encoded)
    raise SystemExit(0 if report['source_model_gate_pass'] else 2)
