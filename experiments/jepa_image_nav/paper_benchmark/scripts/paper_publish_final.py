"""Publish only accepted compact benchmark scores, code and aggregate evidence.

Never copies raw episodes, photographs, prompts, actions, poses or credentials.
Requires the exception-aware final assessment and preserves all older directories.
"""
import hashlib
import json
import re
import shutil
import statistics
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('/root/autodl-tmp/jepa_vln_paper_bench_20261017')
REPO = Path('/root/autodl-tmp/lewm_research/github/LeWM_Navigation')
OUT = REPO/'experiments/jepa_image_nav/paper_benchmark'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


def write(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, indent=2)+'\n'
    if p.exists():
        assert p.read_text() == text, 'Refusing to overwrite differing published file: '+str(p)
    else:
        p.write_text(text)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items() if not str(k).startswith('private_')}
    if isinstance(value, list):
        return [clean(x) for x in value]
    if isinstance(value, str):
        return value.replace('/root/autodl-tmp/', '${RESEARCH_ROOT}/')
    return value


def main():
    certpath = ROOT/'results/compact430_exception_assessment_accepted.private.json'
    cert = read(certpath)
    assert cert['assessment_accepted_under_native_exception_accounting']
    assert cert['pending'] == 0 and cert['native_implementation_exceptions_uncompleted'] == 1
    assert cert['valid_completed_model_runs'] == 1719 and all(c['passed'] for c in cert['checks'])
    stats = read(ROOT/'results/compact430_exception_statistics.json')
    official = read(ROOT/'results/official_gibson_fgprompt_4901/accepted_summary.json')
    assert official['acceptance'] == 'passed' and official['episodes'] == 4901
    for folder in ['scripts', 'protocol', 'manifests', 'results']:
        (OUT/folder).mkdir(parents=True, exist_ok=True)
    write(OUT/'results/compact_statistics.json', stats)
    write(OUT/'results/official_gibson_fgprompt.json', official)
    write(OUT/'results/acceptance.json', dict(
        source_certificate_sha256=sha(certpath), source_certificate_redacted=True,
        all_assigned_cases_accounted=True, assigned=1720, valid_completed=1719,
        native_implementation_exceptions_uncompleted=1, pending=0,
        original_strict_1720_valid_gate_passed=False,
        exception_aware_assessment_passed=True,
        checks=[{k: v for k, v in c.items() if k in ('name', 'passed', 'sha256', 'files')}
                for c in cert['checks']],
        official_gibson_episodes=4901,
        no_exception_imputed_as_model_failure=True))
    write(OUT/'protocol/common_v1.json', read(ROOT/'protocol/common_v1.json'))
    official_cfg = read(ROOT/'results/official_gibson_fgprompt_4901/configuration.json')
    # The serialized base dataset path is overridden by explicitly supplied native episodes.
    write(OUT/'protocol/official_gibson_configuration.json', dict(
        source_configuration_sha256=sha(ROOT/'results/official_gibson_fgprompt_4901/configuration.json'),
        author_runtime=official_cfg['author_runtime'], habitat_sim=official_cfg['habitat_sim'],
        habitat_lab=official_cfg['habitat_lab'], python_hash_seed=0,
        author_config_files_sha256=official_cfg['author_config_files_sha256'],
        goal_sensor=official_cfg['goal_sensor'], success_measure=official_cfg['success_measure'],
        SPL_measure=official_cfg['SPL_measure'], official_hidden_leaderboard_score=False,
        resolution=[128,128], hfov=90, camera_height_m=1.25, agent_radius_m=.1,
        agent_height_m=1.5, allow_sliding=True, forward_m=.25, turn_degrees=30,
        max_actions=500,
        dataset='Original validation episodes explicitly injected into native PointNavDatasetV1; not base-config train path',
        policy=read(ROOT/'results/official_gibson_fgprompt_4901/policy.json')))

    freeze = read(ROOT/'manifests/hssd_val43_compact430_freeze.json')
    write(OUT/'manifests/freeze_summary.json', dict(
        original_manifest_sha256=sha(ROOT/'manifests/hssd_val43_compact430_freeze.json'),
        parent_manifest_sha256=freeze['parent_manifest_sha256'],
        official_scene_split_sha256=freeze['official_scene_split_sha256'],
        source_revision=freeze['source_revision'], selection_seed=freeze['selection_seed'],
        selection_rule=freeze['selection_rule'], selection_inputs=freeze['selection_inputs'],
        scene_ids=freeze['scene_ids'], per_scene_per_task=10, evaluation_only=True,
        panels={k:{x:v for x,v in d.items() if x != 'path'} for k,d in freeze['panels'].items()},
        raw_source_file_count=len(freeze['source_files']),
        raw_sources_and_goal_images_not_distributed=True))
    runtime = read(ROOT/'manifests/compact430_runtime_sources.json')
    source_files = {}
    for path, digest in runtime['sources'].items():
        p = Path(path);assert sha(p) == digest
        dest = OUT/'scripts'/p.name
        write(dest, p.read_text())
        source_files[dest.name] = dict(sha256=digest, origin=p.parent.parent.name)
    extra = '''paper_freeze.py paper_compact_freeze.py paper_archive.py paper_cpu_gate.py
paper_geometry_gate.py paper_fetch_official.py paper_author_config.py paper_official_gibson.py
paper_official_accept.py paper_official_motion_gate.py paper_collision_gate.py
paper_repair_integrity.py paper_compact_acceptance.py paper_exception_audit.py
paper_exception_replay.py paper_exception_stats.py paper_exception_acceptance.py
test_paper_exception_audit.py paper_unigoal_resume_technical.py paper_unigoal_client_gate.py
paper_unigoal_load_gate.py'''.split()
    for name in extra:
        p=ROOT/'tools'/name;write(OUT/'scripts'/name,p.read_text())
        source_files[name]=dict(sha256=sha(p),origin=ROOT.name)
    write(OUT/'scripts/paper_publish_final.py',Path(__file__).read_text())
    source_files['paper_publish_final.py']=dict(sha256=sha(__file__),origin=ROOT.name)
    write(OUT/'scripts/verify_public_results.py',(ROOT/'tools/paper_verify_public_results.py').read_text())
    write(OUT/'manifests/source_hashes.json',dict(
        original_frozen_runtime_manifest_sha256=sha(ROOT/'manifests/compact430_runtime_sources.json'),
        copied_files=source_files, relocation='Files retain original server root constants; no inference source rewritten'))

    # Only scalar episode metrics and identifiers are public; no step-level data.
    allowed = '''episode_id scene_id task status steps goal_stop global_abort success spl
initial_distance_m final_distance_m path_length_m target_fraction_at_end forced_budget_termination
episode_wall_seconds category_supplied'''.split()
    rows=[];diagnostics={};panel_index={};examples=[]
    for task in ['place','instance']:
        panel=[json.loads(x) for x in (ROOT/'panels'/f'hssd_val43_{task}_compact430.jsonl').read_text().splitlines()]
        panel_index[task]=[{k:e[k] for k in ('episode_id','scene_id','distance_bin','detour_bin','view_gap_bin') if k in e}
                           for e in panel]
        for model in (['fgprompt','debit'] if task=='place' else ['goat','unigoal']):
            base=ROOT/'results'/f'{model}_{task}_compact430'
            causes=Counter();collisions=0;forwards=0;actions=0;times=[];slices=defaultdict(list)
            for e in panel:
                f=base/'episodes'/e['episode_id'];p=f/'accepted_summary.json'
                if not p.exists():
                    assert model == 'unigoal'
                    exception=read(base/'implementation_exceptions'/(e['episode_id']+'.json'))
                    # Preserve digest without disclosing failed private trace or prompt inventory.
                    examples.append(dict(episode_id=e['episode_id'], model=model,
                        status='native_implementation_exception_uncompleted_unscored',
                        source_certificate_sha256=sha(base/'implementation_exceptions'/(e['episode_id']+'.json')),
                        diagnosis='Native fixed-map Graph.get_goal to FMM.set_goal IndexError; no policy boundary clipping',
                        score=None, counted_as_navigation_failure=False))
                    continue
                a=read(p);r=a['records'][0]
                assert a['acceptance']=='passed' and r['status']=='complete'
                item={k:r[k] for k in allowed};item.update(model=model,
                    distance_bin=e['distance_bin'],detour_bin=e['detour_bin'],
                    accepted_summary_sha256=sha(p),trace_sha256=a['trace_sha256'])
                rows.append(item);times.append(r['episode_wall_seconds'])
                if r['success']:cause='success'
                elif r['forced_budget_termination']:cause='budget_exhausted_without_success'
                elif r['global_abort']:cause='native_global_abort'
                elif r['goal_stop'] and r['final_distance_m'] >= (.5 if task=='instance' else 1.):
                    cause='voluntary_stop_outside_distance_threshold'
                elif r['goal_stop'] and task=='instance' and r['target_fraction_at_end'] < .025:
                    cause='voluntary_stop_near_approach_but_insufficient_same_instance_pixels'
                else:cause='other_unsuccessful_termination'
                causes[cause]+=1
                for dim in ['distance_bin','detour_bin']:
                    slices[(dim,e[dim])].append(r)
                with (f/(e['episode_id']+'.steps.jsonl')).open() as stream:
                    for line in stream:
                        step=json.loads(line);actions+=1
                        if step['action']==1:
                            forwards+=1;collisions+=int(step['collided'])
            diagnostics[model]=dict(valid_episodes=sum(causes.values()),termination_partition=dict(causes),
                actions=actions,forward_actions=forwards,collided_forward_actions=collisions,
                collided_forward_fraction=collisions/forwards if forwards else None,
                episode_wall_seconds=dict(mean=statistics.mean(times),median=statistics.median(times)),
                timing_scope='Measured adapter episode wall time, includes local model calls but excludes process setup; not isolated GPU policy latency',
                slices=[dict(dimension=dim,bin=label,valid=len(group),successes=sum(x['success'] for x in group),
                             SR=sum(x['success'] for x in group)/len(group),
                             SPL=statistics.mean(x['spl'] for x in group))
                        for (dim,label),group in sorted(slices.items())],
                diagnosis_scope='Terminal symptoms and collision incidence, not a causal attribution to the encoder or wrong-instance identification')
    assert len(rows)==1719 and len(examples)==1
    write(OUT/'results/episode_metrics.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in rows))
    write(OUT/'results/diagnostics.json',diagnostics)
    write(OUT/'results/implementation_exceptions.json',examples)
    write(OUT/'manifests/selected_case_ids.json',panel_index)
    original=read(ROOT/'results/official_gibson_fgprompt_4901/records.json')
    public=[dict(scene_id=r['scene_id'],episode_id=r['episode_id'],steps=r['steps'],
                 final_action=r['final_action'],success=r['native_metrics']['success'],
                 spl=r['native_metrics']['spl']) for r in original]
    assert len(public)==4901
    write(OUT/'results/official_gibson_episode_metrics.jsonl',''.join(json.dumps(r,sort_keys=True)+'\n' for r in public))
    write(OUT/'results/README.md', '''# 评测结果\n\n`compact_statistics.json` 包含43场景的场景加权及回合加权 SR、SPL和场景簇 bootstrap 区间，实例对照仅使用共同有效的429个ID。`episode_metrics.jsonl` 保存1719条紧凑标量记录，不含图像、动作序列、坐标或模型提示。\n\n`diagnostics.json` 汇总终止类型、碰撞、距离及绕行分组和实测回合耗时。终止类型描述失败表现，不直接证明视觉表征或规划模块是原因。`implementation_exceptions.json` 单列原版越界异常，没有性能分数。\n\n`official_gibson_fgprompt.json` 和 `official_gibson_episode_metrics.jsonl` 是原始4901个验证回合的独立结果，不与HSSD混合排名。`acceptance.json` 为服务器私有完整证书的脱敏摘要，保留原证书哈希；不是包含原始轨迹的可公开审计包。\n\n[返回评测入口](../README.md)\n''')
    write(OUT/'protocol/README.md','''# 任务协议\n\n`common_v1.json` 是四组HSSD测试执行前冻结的原始协议。地点和实例分别计分；其 `official_tracks` 字段列举可选标准任务，不表示本目录完成了类别ObjectNav测试。\n\n`official_gibson_configuration.json` 单独记录Gibson原始验证任务、相机、机器人尺寸、滑动开关、原生计分及检查点。该配置与HSSD受控配置不同，不能用两组分数计算域迁移降幅。原生评测直接注入验证回合，不采用底层模板中的默认训练集路径。\n\n[返回评测入口](../README.md)\n''')
    write(OUT/'manifests/README.md','''# 冻结来源\n\n`freeze_summary.json` 保存场景版本、43场景、抽样规则、原始清单和父面板哈希。`selected_case_ids.json` 保存两类430个固定ID及几何分组，不分发原始回合位置与目标图像。\n\n`source_hashes.json` 对应正式测试前固定的24项运行源码及评测辅助代码。`public_files.sha256.json` 校验本目录公开文件。脱敏摘要本身不等同原始私有清单；原始面板需合法数据及服务器完整构建产物重建后按原哈希核对。\n\n[返回评测入口](../README.md)\n''')
    write(OUT/'scripts/README.md','''# 评测代码\n\n冻结与任务接口使用 `paper_freeze.py`、`paper_compact_freeze.py`、`paper_scope.py`、`paper_common.py`、`paper_sim.py`。`paper_eval.py` 与 `paper_unigoal.py` 调用原生预训练模型；其依赖还包括上级[基线代码](../../baselines/README.md)和服务器上的官方仓库、数据及权重。\n\n`paper_unigoal_resume_technical.py` 只负责固定顺序续跑与严格原版异常隔离，不裁剪目标、扩图或新增决策回退。mmap、连接复用与诊断适配须先通过源码中规定的参数、状态、随机数及输出等价检查。\n\n`paper_official_gibson.py` 保留作者原始ImageGoal回合和原生Habitat计分；`paper_official_accept.py` 检查原回合覆盖及原生运动证据。\n\n完整性、独立物理重放、统计和最终验收依次为 `paper_exception_audit.py`、`paper_exception_replay.py`、`paper_exception_stats.py`、`paper_exception_acceptance.py`。原 `paper_compact_acceptance.py` 的430个UniGoal有效回合门槛保留为未通过；异常知情验收不伪造该旧门槛通过。`test_paper_exception_audit.py` 检查异常识别及重放内核不变。\n\n源码保留实际运行服务器路径常量，没有为公开目录改写推理逻辑。闭环重跑需要还原目录、合法资产和各隔离环境；下载此目录并不构成开箱即用安装。仅复核公开数字可运行：\n\n```bash\npython scripts/verify_public_results.py\n```\n\n该命令不加载GPU、不调用API、不重新运行策略，也不能替代私有轨迹的物理重放。\n\n[返回评测入口](../README.md)\n''')
    modelnames=dict(fgprompt='FGPrompt',debit='DEBiT',goat='GOAT',unigoal='UniGoal')
    table=[]
    for m in stats['models']:
        ci=m['scene_cluster_bootstrap95_SR']
        table.append('| '+ ' | '.join([modelnames[m['model']], '地点图像' if m['task']=='place' else '实例图像',
            str(m['valid_episodes'])+'/430',str(m['successes']),f"{m['scene_weighted_SR']*100:.2f}",
            f"{m['scene_weighted_SPL']*100:.2f}",f"{ci[0]*100:.2f}–{ci[1]*100:.2f}"])+ ' |')
    diaglines=[]
    for model,d in diagnostics.items():
        counts='，'.join(f'{k} {v}' for k,v in d['termination_partition'].items())
        diaglines.append(f"- {modelnames[model]}：{counts}；前进动作碰撞占比 {100*d['collided_forward_fraction']:.2f}%，回合耗时中位数 {d['episode_wall_seconds']['median']:.1f}秒。")
    pairs=[]
    for p in stats['paired_comparisons']:
        ci=p['paired_scene_bootstrap95_delta_SR']
        pairs.append(f"- {p['difference']}：共同有效回合 {p['common_valid_episodes']}，场景平均 SR 差 {p['delta_scene_SR']*100:.2f} 个百分点，95%区间 [{ci[0]*100:.2f}, {ci[1]*100:.2f}]。")
    body='''# 图像目标导航性能评测\n\nHSSD官方43个验证场景上的两类图像导航受控测试，以及原始Gibson ImageGoal验证测试。四组共1720个分配回合，1719个有效完成，UniGoal有1个原版固定地图越界异常，单列未完成且不计为导航失败。HSSD结果用于衍生跨域系统对照，不是官方实例导航榜单或语言指令VLN成绩。\n\n## 场景和任务\n\n机器人在带家具的三维室内仿真场景中移动，看到第一视角相机画面，不是俯视平面控制任务。地点图像任务要求到达照片拍摄位置；实例图像任务要求到达照片里同一个物体的有效接近位置并看到该实例。语义ID、目标真实坐标及完整场景只供评测，不输入策略。\n\n每类任务固定430回合，每个官方验证场景10回合，四种方法只运行各自对应任务。目标图像与任务内初态保持一致，冻结后不按测试成绩换回合或调参。原1290回合清单及历史结果、工程异常日志仍保存在服务器。\n\n## 传感器和成功标准\n\nHSSD使用640×480 RGB、79度视场、0.88米相机高度、0.18米机器人半径、1.4米高度，不允许滑动；前进0.25米、转向30度，最多500动作。地点成功要求主动STOP且测地距离小于1米。实例成功要求主动STOP、到冻结接近视点集合距离小于0.5米，并且当前画面同一实例像素占比至少2.5%；SPL使用同一目标集合。实例协议不是官方ObjectNav区域SPL。\n\nFGPrompt、DEBiT均为RGB加目标图像。GOAT另用理想相对GPS与罗盘；UniGoal使用深度和理想里程计，深度物理量程0至5米。各模型原生动作头保留，不能把不同模态的系统结果当作同输入视觉编码器消融。HSSD材质兼容副本的一个贴图坐标修复为各方法共享，几何不变，但不是官方材质逐像素一致。\n\n## HSSD结果\n\n主指标为等场景权重的SR和任务定义SPL，表中数值为百分数。95%区间由43个场景簇bootstrap 10000次计算，固定种子20261017。UniGoal对应429个有效回合，缺失不是随机抽样；另报告430个分配回合的覆盖率。\n\n| 方法 | 任务 | 有效回合 | 成功数 | SR | SPL | SR 95%区间 |\n| --- | --- | --- | --- | --- | --- | --- |\n'''+ '\n'.join(table)+'''\n\n配对比较保留相同有效ID；GOAT完整430回合成绩与共同429回合成绩分别保留。\n\n'''+ '\n'.join(pairs)+'''\n\n## 官方Gibson结果\n\nFGPrompt使用作者公开的14场景4901个原始验证回合、原生目标相机和Habitat成功/SPL计分，成功4083回合。回合加权SR为83.31%，SPL为59.44%；场景加权SR为84.94%，SPL为61.16%。独立审计672329个动作。运行时为Habitat 0.3.1，作者原环境为0.2.2，属于运行时适配的原始任务验证，不是逐位复现或隐藏排行榜提交。\n\n该组使用128×128、90度视场、1.25米相机高度、半径0.1米、高度1.5米机器人和原生滑动，不能把与HSSD的分数差全部解释为场景泛化损失。官方实例任务受场景许可限制，当前HSSD实例结果不替代该标准任务。\n\n## 主要不足\n\nHSSD跨域测试中，各方法的成功率均明显受限。以下是完整轨迹支持的终止表现，而非对单个模块的因果判断；英文分类名与机器可读统计一致。\n\n'''+ '\n'.join(diaglines)+'''\n\n远处主动停止说明停止判定未能稳定对应任务成功；耗尽动作预算表示在给定预算内未完成，但仅凭终止类型不能区分识别、探索、规划或局部控制的主因。终点目标像素不足不直接等于认错物体，遮挡、朝向和相机俯仰也可能导致该结果。距离及绕行分组保存在诊断文件，样本少的分组不能用来确定趋势。\n\nUniGoal原版地图边界越界发生于Graph.get_goal调用FMM.set_goal，索引767超出722边界。该回合全部原始尝试和日志保留，没有扩图、裁剪或添加回退策略；它反映当前原版实现的适用范围，不作为模型导航失败。UniGoal还存在大量串行本地模型调用，回合耗时远大于轻量策略；该时间包含系统开销，不是孤立神经网络推理延迟。\n\n场景级区间不涵盖训练种子变化、非随机缺失或未知预训练场景暴露。当前是固定官方检查点的一次冻结评估，不能证明跨训练种子的稳定优越性。两类图像任务没有语言指令输入，后续JEPA-VLN仍需独立的指令导航基准；这里建立的是图像目标导航对照，不直接验证JEPA改进。\n\n## 文件与复核\n\n- [协议](protocol/README.md)：HSSD共同物理条件、模态边界和Gibson独立配置。\n- [冻结来源](manifests/README.md)：固定回合ID、分层规则和源码哈希。\n- [结果](results/README.md)：统计区间、紧凑逐例指标、终止诊断和异常。\n- [代码](scripts/README.md)：模型接口、几何与计分验收、统计及公开数字复核。\n\n场景、目标图像、权重、逐步私有轨迹、模型提示、API密钥和费用账本不上传。公开标量指标可复算，闭环复现仍需要合法场景和官方权重。\n\n## 上游工作\n\n[HSSD](https://github.com/3dlg-hcvc/hssd)、[Gibson ImageGoal](https://github.com/facebookresearch/image-goal-nav-dataset)、[FGPrompt](https://github.com/XinyuSun/FGPrompt)、[DEBiT](https://github.com/naver/debit)、[GOAT-Bench](https://github.com/facebookresearch/goat-bench)。UniGoal来源与检查点见上级[导航基线](../baselines/navigation_chain/README.md)。\n'''
    write(OUT/'README.md',body)
    # Parent navigation links are surgical; never remove historic experiments.
    for parent, line in [
        (REPO/'README.md','- [图像目标导航冻结评测](experiments/jepa_image_nav/paper_benchmark/README.md)：HSSD43公平对照、原始Gibson标准验证、置信区间与失败表现。'),
        (REPO/'experiments/README.md','- [jepa_image_nav](jepa_image_nav/README.md)：图像地点与实例导航数据、基线及冻结评测。'),
        (OUT.parent/'README.md','- [paper_benchmark](paper_benchmark/README.md)：43个HSSD验证场景的四组受控测试和Gibson原始4901回合验证。')]:
        old=parent.read_text()
        if line not in old:parent.write_text(old.rstrip()+'\n\n'+line+'\n')
    print(json.dumps(dict(public_bundle=str(OUT),valid_scores=len(rows),official_scores=len(public),
                          native_exceptions=len(examples),raw_trajectories_copied=0)))


if __name__=='__main__':
    main()
