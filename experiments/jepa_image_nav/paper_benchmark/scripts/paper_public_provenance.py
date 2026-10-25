"""Export allowlisted deployment metadata without credentials or episode poses."""
import hashlib
import json
from pathlib import Path

ROOT=Path('/root/autodl-tmp/jepa_vln_paper_bench_20261017')
CHAIN=Path('/root/autodl-tmp/jepa_instance_bench_20261016')
OUT=Path('/root/autodl-tmp/lewm_research/github/LeWM_Navigation/experiments/jepa_image_nav/paper_benchmark')


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    content=(json.dumps(d,indent=2,ensure_ascii=False)+'\n') if not isinstance(d,str) else d
    if p.exists():assert p.read_text()==content
    else:p.write_text(content)


def main():
    configurations={}
    for model,task in [('fgprompt','place'),('debit','place'),('goat','instance')]:
        p=ROOT/'results'/f'{model}_{task}_compact430/configuration.json'
        cfg=read(p)
        allow=['model','task','episodes','modalities','policy','panel_sha256','contract_sha256','native_source_sha256']
        configurations[model]={k:cfg[k] for k in allow}
        configurations[model]['source_configuration_sha256']=sha(p)
    keys='''seed env_frame_width env_frame_height frame_width frame_height max_episode_length camera_height
hfov turn_angle min_depth max_depth map_size_cm map_resolution map_size global_width global_height
local_width local_height llm_model vlm_model goal_type category_assisted calibration_changes
goal_category_fallback terminal_execution schema_handling scope'''.split()
    valid=read(ROOT/'results/unigoal_instance_compact430/exception_accounting_accepted.private.json')['episode_ids']['valid']
    cfgs=[];hashes={}
    for eid in valid:
        p=ROOT/'results/unigoal_instance_compact430/episodes'/eid/'configuration.json'
        d=read(p);cfgs.append({k:d[k] for k in keys});hashes[eid]=sha(p)
    assert len(cfgs)==429 and all(d==cfgs[0] for d in cfgs)
    runtime=read(CHAIN/'results/unigoal_llm_runtime_gate.json')
    served=runtime['loaded_models'][0]
    configurations['unigoal']=dict(configuration=cfgs[0], identical_configuration_across_429_valid_cases=True,
        configurations_inventory_sha256=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        local_runtime=dict(model=served['model'],model_digest=served['digest'],
            details=served['details'],context_length=served['context_length'],
            ollama_version=runtime['runtime_version'],
            temperature='Native server/model default; not explicitly overridden',
            scope='Published identity gate and frozen serving configuration; not claimed full-precision author reproduction'),
        depth_note='Physical depth clipped to 0..5m, transport divided by30 and native BEV restored with*30; max_depth30 is normalization, not sensor30m',
        map_exception_preserved=True)
    data=dict(models=configurations, source_versions=dict(
        FGPrompt=dict(repository='XinyuSun/FGPrompt',commit='4430fab50e74c817ab5c7bb28650e569566a234e'),
        DEBiT=dict(repository='naver/debit',git_tree='eecc8f9ba2d24c2e8a1ea9d1ef29bff629e4424f',
                   source_blob_inventory_verified=True,commit_not_inferred_from_tree=True),
        GOAT=dict(repository='Ram81/goat-bench',commit='74c41d19d4a4c3608d1575b512087b5a529aee0e'),
        UniGoal=dict(repository='bagh2178/UniGoal',commit='e2c6d456d53150db507a008250f50c34798daad5')),
        decoding=dict(fgprompt='argmax; torch/numpy initialization seed20261015',
            debit='argmax; torch/numpy initialization seed20261015',
            goat='argmax; torch initialization seed20261016',
            unigoal='torch/numpy reset1 per episode; local server native sampling defaults, not deterministic training-seed replication'),
        pretrained_scene_exposure='Not fully certified across all upstream training sources; evaluation scene split does not prove absence from pretraining',
        no_training_performed=True,weights_not_distributed=True)
    save(OUT/'protocol/model_provenance.json',data)
    save(OUT/'scripts/paper_public_provenance.py',Path(__file__).read_text())
    # Verify the reviewed bundle before intentionally refreshing its unpublished hash inventory.
    old=read(OUT/'manifests/public_files.sha256.json')
    for p,h in old.items():
        if p=='README.md':continue
        assert sha(OUT/p)==h,'Unexpected bundle mutation: '+p
    snapshot=ROOT/'results/public_bundle_before_metadata20261025.private.json'
    save(snapshot,old)
    updated={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*'))
             if p.is_file() and p!=OUT/'manifests/public_files.sha256.json'}
    (OUT/'manifests/public_files.sha256.json').write_text(json.dumps(updated,sort_keys=True,indent=2)+'\n')
    print(json.dumps(dict(metadata_exported=True,valid_UniGoal_configurations=429,all_equal=True,files=len(updated))))


if __name__=='__main__':main()
