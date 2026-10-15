"""Create a complete-panel report from accepted results, never partial scores."""
import json,os
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,PageBreak,Table,TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT=Path(__file__).resolve().parents[1]
IN_REPO=Path(__file__).resolve().parent.name=='scripts'
DATA=ROOT/'results' if IN_REPO else ROOT/'work/baseline_results'
OUT=ROOT/'reports/image_target_baselines.pdf' if IN_REPO else ROOT/'output/pdf/Image-target导航_Baseline筛选与初步评测.pdf'
OUT.parent.mkdir(parents=True,exist_ok=True)
def read(n):return json.loads((DATA/(n+'.json')).read_text(encoding='utf-8'))
accepted=read('final_acceptance');assert accepted['status']=='passed'
results={r['run']:r for r in read('baseline_comparison')}
vis=read('horizontal_visibility_audit');lat=read('isolated_latency');camera=read('camera_extrinsics_audit');slices={r['run']:r['groups'] for r in read('constraint_slices')['runs']}
names=['fgprompt_gibson_val_place','debit_native10_gibson_val_place','fgprompt_hssd_test_place','debit_native10_hssd_test_place','fgprompt_hssd_test_instance','debit_native10_hssd_test_instance']
assert all(results[n]['complete_frozen_panel'] for n in names+['lg_frontier_hssd_test_instance'])
pdfmetrics.registerFont(TTFont('Song',os.environ.get('SIMSUN_FONT','C:/Windows/Fonts/simsun.ttc'),subfontIndex=0))
BLUE=colors.HexColor('#173E59');MUTED=colors.HexColor('#526575');PALE=colors.HexColor('#EAF1F5')
BODY=ParagraphStyle('body',fontName='Song',fontSize=10.4,leading=17,spaceAfter=8,firstLineIndent=0,wordWrap='CJK')
H1=ParagraphStyle('h1',parent=BODY,fontSize=18,leading=25,textColor=BLUE,spaceAfter=16)
H2=ParagraphStyle('h2',parent=BODY,fontSize=12.5,leading=19,textColor=BLUE,spaceAfter=8,spaceBefore=8)
SMALL=ParagraphStyle('small',parent=BODY,fontSize=9,leading=14,textColor=MUTED,spaceAfter=5)
CELL=ParagraphStyle('cell',parent=BODY,fontSize=9.1,leading=14,spaceAfter=0)
TITLE=ParagraphStyle('title',parent=BODY,fontSize=24,leading=34,alignment=TA_CENTER,textColor=BLUE)
story=[]
def p(s,style=BODY):return Paragraph(s,style)
def add(s,style=BODY):story.append(p(s,style))
def page(title):
    if story:story.append(PageBreak())
    add(title,H1)
def link(url,label):return '<link href="'+escape(url)+'" color="#245E86">'+label+'</link>'
def table(rows,widths):
    obj=Table([[p(str(v),CELL) for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
    obj.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),PALE),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#8BA6B8')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#CDD9E1')),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7)]))
    story.append(obj);story.append(Spacer(1,10))
def pct(x):return f'{100*x:.1f}%'
def count(r,key):return int(round(r[key]*r['episodes']))
def outcome(r,key):return r['outcomes'].get(key,0)
labels=['FGPrompt / Gibson地点','DEBiT-S 10° / Gibson地点','FGPrompt / HSSD地点','DEBiT-S 10° / HSSD地点','FGPrompt迁移 / HSSD实例','DEBiT-S迁移 / HSSD实例']
lg=results['lg_frontier_hssd_test_instance'];fg=results['fgprompt_hssd_test_instance'];de=results['debit_native10_hssd_test_instance']

story.append(Spacer(1,175));add('Image-target 导航<br/>Baseline 筛选与性能评测',TITLE)
story.append(Spacer(1,20));add('地点图像目标与同一物体实例目标',ParagraphStyle('sub',parent=BODY,alignment=TA_CENTER,fontSize=13))
add('公开预训练模型 · 三维仿真闭环 · JEPA 性能对照',ParagraphStyle('sub2',parent=SMALL,alignment=TA_CENTER,fontSize=11))
story.append(Spacer(1,28));add('实验与资料核查：2026年10月16日',ParagraphStyle('date',parent=SMALL,alignment=TA_CENTER))

page('1. 任务与完整主结果')
add('研究对象是给定一张目标图像的第一视角导航，不天然包含语言指令。地点任务要求回到照片拍摄位置；实例任务要求找到照片中的同一个物体，而不是任意同类别物体。两者独立评分。')
add('RGB主评测完成1,760次模型-回合执行，覆盖880个固定任务回合；另完成280次转向接口对照和300次RGB-D辅助系统测试。策略直接使用公开预训练权重，没有训练或微调。')
rows=[['预训练策略与任务','回合','SR','SPL','SR 95%区间']]
for name,label in zip(names,labels):
    r=results[name];lo,hi=r['SR_scene_bootstrap_95'];rows.append([label,r['episodes'],pct(r['SR']),f"{r['SPL']:.3f}",pct(lo)+' - '+pct(hi)])
table(rows,[200,43,61,60,147])
add('结果的含义',H2)
add('Gibson是真实扫描场景，HSSD是合成室内场景。跨数据集差距同时涉及视觉、布局和任务难度，不单独归因于编码器。实例两行使用地点导航检查点迁移，不能代表实例专用导航模型的能力上限。')
add('FGPrompt使用30°转向，DEBiT使用原生10°转向，均为500次决策。它们不是同一物理转向预算下的算法排名。当前结果属于本项目派生协议下的预训练迁移评测，不是原论文表格复现。')
add('区间按场景均值进行5,000次bootstrap；每场景20回合并按距离与绕行分层，分数不代表原始数据的自然频率总体。')

page('2. 与任务直接对应的开源基线')
table([['方法','架构与主要作用','资源及测试位置'],
 ['FGPrompt\nNeurIPS 2023','目标细节作为提示进入观测编码器，循环策略直接预测离散动作。对应地点图像导航。','官方Early-fusion-r9权重；Gibson地点、HSSD地点与实例迁移均完成。'],
 ['DEBiT\nICLR 2024','单目空间编码与双目Transformer结合，利用图像对应关系预训练，再接循环导航策略。','官方DEBiT-S ImageNav权重；三个面板完成。不是论文实例专用检查点。'],
 ['Mod-IIN\nICCV 2023','探索、实例重识别、匹配点定位和局部执行分别建模。适合模块化导航参照。','本项目LightGlue-frontier为组件适配，不是该论文官方系统复现。'],
 ['IEVE\nCVPR 2024','对候选实例先靠近再确认，在探索、验证、利用之间切换，直接处理实例身份。','官方代码和预训练入口公开；完整分割、类别与速度控制链未在本项目测试。'],
 ['UniGoal\nCVPR 2025','目标与在线三维场景采用图表示，结合匹配和视觉语言推理进行搜索、验证。','官方支持实例图像目标，可用本地VLM；完整系统未在本项目测试。']],[92,211,208])
add('基线组合与研究参考',H2)
add('FGPrompt提供轻量RGB循环策略参照，DEBiT提供对应关系表征参照。LightGlue-frontier检验局部匹配与显式探索的组合，但不是实例专用方法的替代品。后续实例系统比较应增加完整IEVE或UniGoal，保持输入和执行预算分轨报告。')
add('官方来源：'+link('https://github.com/XinyuSun/FGPrompt','FGPrompt')+' / '+link('https://github.com/naver/debit','DEBiT')+' / '+link('https://jacobkrantz.github.io/modular_iin','Mod-IIN')+' / '+link('https://github.com/XiaohanLei/IEVE','IEVE')+' / '+link('https://github.com/bagh2178/UniGoal','UniGoal'),SMALL)

page('3. 路线模型与部署参考')
table([['方法','适合借鉴的部分','适用边界'],
 ['PIG-Nav','公开预训练的图像目标局部路线，早期融合与导航辅助任务。','需校准waypoints、距离尺度和控制频率；不是实例身份确认系统，未在本项目闭环测试。'],
 ['ViNT / NoMaD','跨机器人局部视觉运动、路线生成与探索。','有预采集拓扑图的路线跟随与未知地图搜索应单列；本项目未接入。'],
 ['NavDP / X-NavDP','RGB-D局部路线与避障、扩散轨迹筛选。','NavDP核查时权重入口要求申请；本项目未申请。X-NavDP公开PointNav不能直接替代ImageGoal。'],
 ['IGL-Nav','在在线三维表示中定位目标相机，适合精确停靠参考。','核查时官方仍标注代码即将发布，未作为可运行基线。'],
 ['Jev-ObjNav','区域图、候选动作验证、目标搜索与进展记忆。','原接口使用目标名称，Jev通过外部服务接入；类别正确不等于同一实例，本项目未调用该服务。'],
 ['JEPLO','LiDAR与本体感知驱动的四足运动执行。','不负责图像目标搜索或识别。JEPLO-Dataset为8段真机记录，不是ImageGoal仿真任务。']],[93,191,227])
add('Jev可提供高层探索组织参考，JEPLO可承接低层运动命令。目标实例识别、空间定位和STOP仍需导航系统完成；两者不直接进入本项目ImageGoal成功率排名。')
add('来源：'+link('https://github.com/zpschang/PIG-Nav','PIG-Nav')+' / '+link('https://github.com/robodhruv/visualnav-transformer','ViNT与NoMaD')+' / '+link('https://github.com/InternRobotics/NavDP','NavDP')+' / '+link('https://github.com/GWxuan/IGL-Nav','IGL-Nav')+' / '+link('https://github.com/owlet914/jev-objnav','Jev-ObjNav')+' / '+link('https://github.com/ASIG-X/JEPLO','JEPLO')+' / '+link('https://huggingface.co/datasets/ASIG-Lab/JEPLO-Dataset','JEPLO-Dataset'),SMALL)

page('4. 仿真环境与输入接口')
add('环境是三维室内场景中的第一视角导航，不是二维俯视图。机器人通过简化刚体与导航网格执行平移和转向；这些结果不能评估四足跌倒、上下楼梯或真机接触动力学。')
table([['条件','设置'],['场景与回合','Gibson原作者14个验证场景；HSSD固定15个本地测试场景。每场景20回合，两种任务独立面板。'],
 ['相机与身体','观测RGB 256×256、HFOV 90°、高度1.25米；身体半径0.18米、高度1.4米。带俯仰目标图像的真实光心见第7节。'],
 ['动作与预算','STOP、前进0.25米、左右转向；FGPrompt 30°，DEBiT 10°；最多500决策，选择最大logit动作。'],
 ['地点成功','明确STOP，并且到目标照片拍摄位置的测地距离小于1米。'],
 ['实例成功','明确STOP，距任一已验证接近视点的测地距离小于0.5米，且当前相机正确实例像素占比至少2.5%。'],
 ['辅助指标','SPL、SoftSPL、碰撞前进率、结果分类；地点另看0.25米且朝向差小于15°的严格位姿成功。']],[102,409])
add('策略与评分隔离',H2)
add('RGB策略仅接收目标图像、当前图像和自身动作历史。目标位置、实例ID、语义图和全局网格只用于评分。RGB-D辅助系统额外获得深度及由仿真真值计算的理想相对里程计，独立于RGB主轨报告。')
add('计算环境',H2)
add('仿真使用独立Python 3.9 / Habitat-Sim 0.3.3；策略使用PyTorch 2.8.0 / CUDA 12.8与RTX 5090。官方网络计算类和函数体保留，替换训练注册及无关传感器依赖，全部策略权重键严格匹配。FGPrompt输入缩放至128，DEBiT单目224、双目112。')
add('原服务器GPU故障后，保留已完成场景并在健康服务器续跑。数据划分、固定回合、目标图像、检查点和评分阈值不变；新增场景级锁仅保护结果写入，不改变动作计算。')

page('5. 长程路线、接口与精确停止')
cols=[results[n] for n in ['fgprompt_gibson_val_place','debit_native10_gibson_val_place','fgprompt_hssd_test_place','debit_native10_hssd_test_place']]
rows=[['距离','FG Gibson','DEBiT Gibson','FG HSSD','DEBiT HSSD']]
for key,label in [('short','1.5至3米以下'),('medium','3至5米以下'),('long','5至10米以下'),('extra','10米及以上')]:
    rows.append([label]+[pct(r['by_distance_bin'][key]['SR'])+f"（{r['by_distance_bin'][key]['episodes']}）" for r in cols])
table(rows,[107,101,101,101,101])
add('括号为回合数。同一数据面板的模型共享回合，不同数据面板不共享地图和路线；跨数据差距不能单独解释为纹理或几何影响。长程和绕行切片是固定的后续比较条件。')
table([['地点面板','直接路线SR','绕行路线SR','严格位姿SR']]+[[label,pct(results[n]['by_detour_bin']['direct']['SR']),pct(results[n]['by_detour_bin']['detour']['SR']),pct(results[n]['strict_pose_success_rate'])] for n,label in zip(['fgprompt_gibson_val_place','debit_native10_gibson_val_place','fgprompt_hssd_test_place','debit_native10_hssd_test_place'],['FG / Gibson','DEBiT / Gibson','FG / HSSD','DEBiT / HSSD'])],[170,110,110,121])
add('控制接口配对诊断',H2)
add('相同DEBiT-S权重、相同280个Gibson回合、相同500次决策下，30°失配SR为21.4%，10°对齐为66.4%，相差45.0个百分点，场景bootstrap区间为37.9至52.1个百分点。它是控制接口效应，不是算法收益。')
add('宽松到达不等于精确停靠。原策略未必优化目标相机朝向，因此严格位姿分数应作为独立任务要求，不以宽松SR代替。')

page('6. 实例目标与失败表现')
table([['实例面板','SR / SPL','曾清楚看到目标','曾进入可见成功区','其他收录实例处STOP']]+[
 [label,pct(r['SR'])+f" / {r['SPL']:.3f}",f"{count(r,'target_visible_ever_rate')}/300",f"{count(r,'reached_success_region_rate')}/300",f"{count(r,'wrong_instance_stop_rate_selected_instances_only')}/300"]
 for label,r in [('FGPrompt迁移',fg),('DEBiT-S迁移',de),('LightGlue-frontier',lg)]],[105,105,98,102,101])
add('前两者为RGB地点检查点迁移；第三者额外使用RGB-D和理想相对里程计，是本项目适配系统，不是Mod-IIN、IEVE或UniGoal的官方复现。三行不能按同一输入条件排名。')
table([['可观测结果','FG迁移','DEBiT迁移','匹配与探索']]+[[label]+[outcome(r,key) for r in [fg,de,lg]] for key,label in [
 ('success','正确STOP成功'),('premature_stop','成功区外提前STOP'),('reached_success_region_but_failed_to_stop','曾进入成功区但未正确STOP'),
 ('wrong_instance_stop','其他收录同类实例处STOP'),('collision_dominated_timeout','碰撞主导的超时'),('search_or_localization_timeout','其他搜索或定位超时')]], [238,91,91,91])
add('如何解释这些失败',H2)
add('到达成功区域却未正确停止，说明搜索、接近与STOP应分开诊断；从未看到目标则仍涉及探索与路线。错误实例STOP只覆盖其他拥有接近标签的同类目标，不能当作所有家具的身份混淆率。结果标签不是网络内部失败原因的因果证明。')
add('匹配辅助系统的限制',H2)
add('整张目标照片的匹配会受背景、视角变化和重复家具影响；中心区域筛选不是目标分割。深度投影、占据图、frontier选择和停止阈值都可能限制闭环表现。该系统提供组件级对照，不能据此评价完整实例导航方法的上限。')

page('7. 相机可见性与资产泛化')
add('水平相机核查',H2)
add(f"原目标标注的position为机器人原点，height为旋转前的局部相机偏移。带俯仰时实际光心也会移动，不能直接当作固定世界高度。已为168个场景、{camera['target_images']:,}张目标图像补充真实光心与旋转外参，并用实时传感器状态核对；图像和评分未变。")
add(f"测试场景的{vis['target_instances']}个已收录实例中，{vis['without_a_verified_yaw_only_approach_position']}个在所有已保存接近位置上，水平相机均未满足2.5%像素占比；对应固定实例面板中的{vis['panel_episodes_without_verified_yaw_only_approach']}个回合。核查把相机俯仰设为0，其他场景与实例配置保持一致。")
add('这说明目标照片与水平动作接口存在需要单列的约束，但不证明目标不可到达：已保存位置并不覆盖所有可导航位置。原始300回合及评分全部保留，下表是事后诊断切片，不替代主结果。')
rows=[['实例策略','水平接近位置已验证','水平接近位置未验证']]
for n,label in [('fgprompt_hssd_test_instance','FG迁移'),('debit_native10_hssd_test_instance','DEBiT迁移'),('lg_frontier_hssd_test_instance','匹配与探索')]:
    g=slices[n];rows.append([label]+[pct(g[k]['SR'])+f"（{g[k]['episodes']}）" for k in ['yaw_stored_approach_verified','yaw_stored_approach_unverified']])
table(rows,[143,184,184])
add('资产复用与场景隔离',H2)
add('319个测试目标中，120个复用了训练目标的资产，173个在本场景还存在同资产的其他已收录实例。场景隔离不等于资产隔离，同资产也不等于目标照片完全相同。')
rows=[['实例策略','训练目标中出现过该资产','训练目标中未出现该资产']]
for n,label in [('fgprompt_hssd_test_instance','FG迁移'),('debit_native10_hssd_test_instance','DEBiT迁移'),('lg_frontier_hssd_test_instance','匹配与探索')]:
    g=slices[n];rows.append([label]+[pct(g[k]['SR'])+f"（{g[k]['episodes']}）" for k in ['asset_seen_in_selected_train_targets','asset_not_seen_in_selected_train_targets']])
table(rows,[143,184,184])
add('这里的“出现过”只检查已选训练目标，不覆盖训练场景中的全部家具。模型预训练可能使用其他数据；上述切片不是严格的模型训练曝光证明。',SMALL)

page('8. 推理开销、验收与研究对照')
add('独占GPU推理测量',H2)
table([['策略','网络均值 / P95','RPC均值 / P95','测量次数']]+[[r['model'],f"{r['network_forward']['mean_ms']:.2f} / {r['network_forward']['p95_ms']:.2f} ms",f"{r['rpc_wall']['mean_ms']:.2f} / {r['rpc_wall']['p95_ms']:.2f} ms",r['network_forward']['n']] for r in lat['runs']],[105,154,164,88])
add('测量时无其他CUDA计算作业。18组固定面板起点图像，每模型预热30次，每组重复10次。网络计时不含图像预处理；RPC计时包含数组加载、预处理与通信，不含场景渲染、目标重置和准备图片。静态重放不是闭环实时控制，也不是等时间预算导航。')
add('完整性验收',H2)
add('8个完整模型-面板组合共2,340次执行，逐场景核对固定回合ID、面板SHA-256、动作预算、转向配置、回合数、逐步轨迹长度、成功条件与SPL。RGB主轨1,760次，接口对照280次，RGB-D辅助轨300次；319个实例完成水平可见性核查。启动错误没有转换成导航失败。')
add('JEPA扩展的固定对照',H2)
add('先比较历史表征融合，再测试动作条件潜变量预测能否改善进展、可见性和候选确认。保持感知与执行接口不变，设置原策略、等参数静态编码器、仅加记忆、仅改STOP、潜预测动作筛选与规划细化。实例表征需要保留身份细节，不能为了不变性抹去实例差异。')
add('完整实例专用方法和等时间预算闭环实验仍是进一步比较项目，不包含在已验收分数内。JEPA不能凭空恢复未观察地图，也不能仅凭完全相同外观保证实例唯一。')
add('结论',H2)
add('当前固定面板建立了可重跑的预训练RGB对照和独立RGB-D组件参照。重点问题是跨分布目标对齐、长程探索、实例主动确认与停止决策；后续改进应逐项消融，并区分模型能力、相机可见性和控制接口的影响。')
add(link('https://github.com/ZhuShanzhe/LeWM_Navigation/tree/jepa-image-nav-datasets-20261014/experiments/jepa_image_nav/baselines','项目代码、聚合结果与验收记录'),SMALL)

def footer(canvas,doc):
    w,h=doc.pagesize;canvas.setStrokeColor(colors.HexColor('#C9D7E0'));canvas.setLineWidth(.4)
    canvas.line(42,40,w-42,40);canvas.setFont('Song',8.5);canvas.setFillColor(MUTED)
    canvas.drawString(42,27,'Image-target 导航 | 预训练基线与闭环评测');canvas.drawRightString(w-42,27,str(doc.page))
doc=SimpleDocTemplate(str(OUT),pagesize=(595.28,841.89),leftMargin=42,rightMargin=42,topMargin=42,bottomMargin=53,
                      title='Image-target 导航 Baseline 筛选与性能评测',author='JEPA Navigation Research')
doc.build(story,onFirstPage=footer,onLaterPages=footer);print(OUT)
