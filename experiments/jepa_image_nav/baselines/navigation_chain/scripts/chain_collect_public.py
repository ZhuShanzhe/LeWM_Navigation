"""Curate project-owned scripts and aggregate evidence; exclude raw/private data."""
from collections import Counter
from datetime import datetime,timezone
import difflib,hashlib,json,re,shutil,subprocess
from pathlib import Path
ROOT=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
REPO=Path('/root/autodl-tmp/lewm_research/github/LeWM_Navigation')
OUT=REPO/'experiments/jepa_image_nav/baselines/navigation_chain'
for folder in ['scripts','results','literature','patches']:(OUT/folder).mkdir(parents=True,exist_ok=True)
def write(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False))
mapping={
 'goat_instance_comparison.json':'results/goat_instance_comparison.json',
 'goat_category_accepted.json':'results/goat_category_hssd300_v3/accepted_summary.json',
 'category_numerical_diagnostic.json':'results/goat_category_hssd300_v3/numerical_diagnostic.json',
 'goat_native_geometry_rejected.json':'results/goat_native_hssd300/rejected_summary.json',
 'objectnav_dataset_manifest.json':'datasets/hssd_objectnav_v3/manifest.json',
 'jeplo_motion_accepted.json':'results/jeplo_native_motion_validated/accepted_summary.json',
 'eb_navigation_pilot_accepted.json':'results/eb_navigation_qwen25vl7b/accepted_pilot_summary.json'}
for dest,source in mapping.items():
    path=ROOT/source
    assert path.exists(),source
    shutil.copy2(path,OUT/'results'/dest)
script_names='''instance_fetch_sources.py instance_fetch_assets.py instance_fetch_extensions.py
instance_download_models.py instance_checkpoint_audit.py instance_cache_audit.py
instance_policy_goat.py instance_eval_goat.py instance_sensor_gate.py instance_build_objectnav_v2.py
chain_goat_acceptance.py chain_motion_replay.py chain_goat_objectnav.py chain_finalize_objectnav.py
chain_category_acceptance.py chain_category_numeric_diagnostic.py chain_reject_native_geometry.py
chain_download_runtime.py chain_provision_eb.sh chain_ai2thor_asset.py chain_verify_model_assets.py
chain_eb_navigation.py chain_run_eb_full.sh chain_eb_acceptance.py chain_jeplo_batch.py
jeplo_native_smoke.py chain_motion_acceptance.py chain_setup_unigoal.py chain_unigoal_build.sh
chain_unigoal_import.py chain_unigoal_sim.py chain_unigoal_run.py chain_download_bert.py
chain_prefetch_unigoal_matcher.py chain_ollama_compatible.py chain_unigoal_runtime_gate.py
chain_unigoal_ready_smoke.py chain_download_pixi.py pixi_ros.toml chain_build_jev.sh
chain_prepare_jev_models.py chain_provision_jev_vision.sh chain_download_jev_blip.py
chain_jev_native_gate.py chain_convert_jev_yolo.py chain_collect_public.py'''.split()
for name in script_names:
    path=ROOT/'runtime/ros_workspace/pixi.toml' if name=='pixi_ros.toml' else ROOT/'tools'/name
    assert path.is_file(),name
    shutil.copy2(path,OUT/'scripts'/name)
shutil.copy2(ROOT/'runtime/ros_workspace/pixi.lock',OUT/'scripts/pixi_ros.lock')
patches=[
 ('unigoal_slam_classes.patch','results/source_snapshots/UniGoal_slam_classes.original.py','repos/UniGoal/src/graph/utils/slam_classes.py','src/graph/utils/slam_classes.py'),
 ('unigoal_llm.patch','results/source_snapshots/UniGoal_llm_original.py','repos/UniGoal/src/utils/llm.py','src/utils/llm.py'),
 ('groundingdino_torch_dispatch.patch','results/source_snapshots/ms_deform_attn_cuda.original.cu','repos/GroundingDINO/groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu','groundingdino/models/GroundingDINO/csrc/MsDeformAttn/ms_deform_attn_cuda.cu'),
 ('jev_geo_utils.patch','results/source_snapshots/jev_geo_utils.original.hpp','repos/jev-objnav/jev_obj/src/planner/trajectory_manager/include/gcopter/geo_utils.hpp','jev_obj/src/planner/trajectory_manager/include/gcopter/geo_utils.hpp')]
patch_records=[]
for name,before,after,label in patches:
    a=ROOT/before;b=ROOT/after
    diff=subprocess.run(['git','diff','--no-index','--no-ext-diff','--',str(a),str(b)],capture_output=True,text=True)
    assert diff.returncode==1,diff.stderr
    lines=diff.stdout.splitlines(True)
    lines[0]='diff --git a/'+label+' b/'+label+'\n'
    lines=[('--- a/'+label+'\n') if line.startswith('--- ') else
        ('+++ b/'+label+'\n') if line.startswith('+++ ') else line for line in lines]
    delta=''.join(lines)
    assert delta,name
    (OUT/'patches'/name).write_text(delta)
    patch_records.append({'file':name,'original_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),
        'adapted_sha256':hashlib.sha256(b.read_bytes()).hexdigest()})
sources=[]
for name in ['goat-bench','EmbodiedBench','UniGoal','jev-objnav','GroundingDINO','MobileSAM','yolov7','CLIP','vlfm','GC-VLN']:
    folder=ROOT/'repos'/name
    try:sha=subprocess.check_output(['git','-C',str(folder),'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        meta=json.loads((ROOT/'results'/(name+'_download.json')).read_text());sha=meta['commit']
    sources.append({'name':name,'commit':sha})
assets=[]
for name in ['goat_checkpoints__sense_act_nn_monolithic__ckpt_best.pth.download.json',
    'clip_RN50_openai.safetensors.download.json','sam_vit_h_4b8939.pth.download.json',
    'groundingdino_swint_ogc.pth.download.json','yolov7-e6e.pt.download.json']:
    data=json.loads((ROOT/'results'/name).read_text())
    assets.append({k:data[k] for k in ['filename','url','bytes','sha256'] if k in data})
gates={}
for name in ['goat_sensor_gate','qwen_integrity_gate','ai2thor_runtime','bert_assets','unigoal_module_import',
    'unigoal_matcher_assets','jev_native_build_gate','jev_blip_assets','jev_yolo_safe_conversion',
    'ollama_compatible_runtime','unigoal_llm_runtime_gate','qwen_assets']:
    path=ROOT/'results'/(name+'.json')
    if path.exists():gates[name]=json.loads(path.read_text())
for name,relative in [('unigoal_sensor','unigoal_sim_gate_v2/simulator_gate.json'),
    ('unigoal_cuda','unigoal_full_chain_smoke1_v4/cuda_operator_gate.json'),
    ('eb_sensor','eb_navigation_qwen25vl7b/simulator_gate.json')]:
    gates[name]=json.loads((ROOT/'results'/relative).read_text())
write(OUT/'results/resource_audit.json',{'sources':sources,'assets':assets,'gates':gates,'source_patches':patch_records})
eb=[]
for subset in ['base','common_sense','complex_instruction','visual_appearance','long_horizon']:
    for path in (ROOT/'results/eb_navigation_qwen25vl7b'/subset).glob('[0-9][0-9][0-9].json'):
        row=json.loads(path.read_text());eb.append({'subset':subset,'status':row['status']})
smokes=[]
for folder in sorted((ROOT/'results').glob('unigoal_full_chain_smoke*')):
    failure=folder/'runtime_failure.json';records=folder/'records.json'
    smokes.append({'run':folder.name,'recorded_episodes':len(json.loads(records.read_text())) if records.exists() else 0,
        'runtime_failure':json.loads(failure.read_text()) if failure.exists() else None})
state={'captured_utc':datetime.now(timezone.utc).isoformat(),'all_requested_benchmarks_complete':False,
    'goat_accepted_instance_executions':600,'goat_accepted_category_executions':300,'jeplo_accepted_runs':15,
    'eb_navigation':{'required':300,'record_status_counts':dict(Counter(x['status'] for x in eb)),
        'full_acceptance_exists':(ROOT/'results/eb_navigation_qwen25vl7b/accepted_summary.json').exists()},
    'unigoal_smokes':smokes,'jev_closed_loop_accepted':False,
    'pending':['EB-Navigation full 300 acceptance','UniGoal native full-chain smoke and instance benchmark',
        'Jev native vision services plus official API and ROS/Habitat closed loop']}
write(OUT/'results/execution_status.json',state)
for path in OUT.rglob('*'):
    if not path.is_file():continue
    assert path.suffix in {'.md','.py','.sh','.json','.toml','.patch','.lock'},path
    assert path.stat().st_size<2_000_000,path
    text=path.read_text()
    assert not re.search(r'sk-or-v1-[A-Za-z0-9]{20,}|-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{20,}',text),path
for folder in [OUT,*[p for p in OUT.rglob('*') if p.is_dir()]]:assert (folder/'README.md').is_file(),folder
print(json.dumps({'files':sum(p.is_file() for p in OUT.rglob('*')),'status':state},indent=2))
