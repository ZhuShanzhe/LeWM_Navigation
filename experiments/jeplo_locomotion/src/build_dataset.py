"""Deterministic, independently designed Go2 closed-loop evaluation scenario dataset.

Terrain ideas are cited in README; geometry and parameter ranges are our protocol,
not a port of a paper's benchmark. Robot assets retain upstream GPL-3.0 licensing.
"""
import os
import argparse, collections, hashlib, json, math
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
ROOT=Path(os.environ.get('JEPLO_WORKSPACE', '/root/autodl-tmp/jeplo_eval_20260928')).expanduser().resolve()
ROBOT=Path(os.environ.get('JEPLO_REPO', str(ROOT/'JEPLO')))/'deployment/unitree_mujoco/unitree_robots/go2/go2.xml'
FAMILIES=['flat','box','stairs_up','stairs_down','slope_up','slope_down','gap',
          'rough','hurdles','stepping_stones','narrow_bridge','low_clearance',
          'narrow_passage','mixed_course','long_course']
SPLITS={'development':(11000,2),'validation':(22000,3),'test':(33000,10)}
CONDITIONS=['clean','mask0.2','mask0.4','mask0.6','mask0.8','blob','cross','cut','stripe',
            'delay100','delay300','noise5cm','sparse1','blackout1s','blind','reset_memory',
            'friction_low','mass_plus20','push_lateral','speed_slow','speed_fast']

def generate(family,level,seed):
    rng=np.random.default_rng(seed);root=ET.parse(ROBOT).getroot()
    root.find('compiler').set('meshdir',str(ROBOT.parent/'assets'))
    root.find('option').set('timestep','.002')
    world=root.find('worldbody');asset=root.find('asset');primitives=[]
    ET.SubElement(world,'light',pos='0 0 8',dir='0 0 -1')
    visual=ET.SubElement(root,'visual');ET.SubElement(visual,'global',offwidth='640',offheight='480')
    def box(name,x,y,z,sx,sy,sz):
        assert min(sx,sy,sz)>0
        ET.SubElement(world,'geom',name=name,type='box',pos=f'{x} {y} {z}',size=f'{sx} {sy} {sz}',rgba='.50 .59 .70 1')
        primitives.append({'type':'box','name':name,'x':x,'y':y,'z':z,'sx':sx,'sy':sy,'sz':sz})
    pit=family in ['gap','stepping_stones','narrow_bridge']
    ET.SubElement(world,'geom',name='floor',type='plane',pos=f'0 0 {-1.2 if pit else 0}',size='0 0 .05',friction='2 .005 .0001',rgba='.25 .29 .33 1')
    goal=6.;top=0.;start=0.;p={'jitter_factor':float(rng.uniform(.94,1.06))};j=p['jitter_factor']
    if family=='box':
        h=[.2,.35,.5][level]*j;box('box',1.85,0,h/2,.85,1.5,h/2)
        goal=1.65;top=h;p.update(height=h)
    elif family in ['stairs_up','stairs_down']:
        h=[.10,.16,.22][level]*j;run=float(rng.uniform(.28,.36));n=6
        for k in range(n):
            height=(k+1 if family=='stairs_up' else n-k)*h
            box('step_'+str(k),1+run*(k+.5),0,height/2,run/2,2,height/2)
        if family=='stairs_up':
            top=n*h;goal=1+n*run+.6;box('landing',1+n*run+1.5,0,top/2,1.5,2,top/2)
        else:
            start=(n+1)*h;box('start_platform',-1,0,start/2,2,2,start/2);goal=1+n*run+.8
        p.update(step_height=h,step_run=run,steps=n)
    elif family in ['slope_up','slope_down']:
        angle=[10,20,30][level]*j;length=3.5;h=length*math.tan(math.radians(angle))
        z0,z1=(0,h) if family=='slope_up' else (h,0)
        vertices=[[1,-2,-.1],[1,2,-.1],[4.5,2,-.1],[4.5,-2,-.1],
                  [1,-2,z0],[1,2,z0],[4.5,2,z1],[4.5,-2,z1]]
        ET.SubElement(asset,'mesh',name='ramp_mesh',vertex=' '.join(str(v) for xyz in vertices for v in xyz))
        ET.SubElement(world,'geom',name='ramp',type='mesh',mesh='ramp_mesh',rgba='.50 .59 .70 1')
        primitives.append({'type':'ramp','name':'ramp','x':2.75,'y':0,'sx':1.75,'sy':2,'z0':z0,'z1':z1})
        if family=='slope_up':top=h;box('landing',6.5,0,h/2,2,2,h/2)
        else:start=h;box('start_platform',-1,0,h/2,2,2,h/2)
        p.update(slope_degrees=angle)
    elif family=='gap':
        width=[.2,.45,.7][level]*j
        box('takeoff',-1,0,-.1,2,2,.1);box('landing',5+width/2,0,-.1,4-width/2,2,.1)
        p.update(gap_width=width)
    elif family=='rough':
        amp=[.025,.06,.12][level]*j
        for i in range(18):
            for k in range(12):
                h=float(rng.uniform(.005,amp));box(f'rough_{i}_{k}',1+.25*(i+.5),-.25*6+.25*(k+.5),h/2,.125,.125,h/2)
        p.update(roughness_max=amp)
    elif family=='hurdles':
        h=[.08,.16,.28][level]*j
        for k in range(4):box('hurdle_'+str(k),1.3+1.1*k,0,h/2,.08,1.5,h/2)
        p.update(hurdle_height=h)
    elif family=='stepping_stones':
        gap=[.08,.16,.25][level]*j;length=.45
        box('start_platform',-1,0,-.1,2,2,.1)
        for k in range(7):box('stone_'+str(k),1+gap+length/2+k*(length+gap),float(rng.uniform(-.07,.07)),.04,length/2,.4,.14)
        end=1+7*(length+gap);box('landing',end+1.5,0,-.1,1.5,2,.1);goal=end+.7
        p.update(stone_gap=gap,stone_length=length,stone_width=.8)
    elif family=='narrow_bridge':
        width=[.65,.45,.3][level]*j
        box('start_platform',-1,0,-.1,2,2,.1);box('bridge',3,0,-.1,2,width/2,.1);box('landing',7,0,-.1,2,2,.1)
        p.update(bridge_width=width)
    elif family=='low_clearance':
        clearance=[.55,.42,.30][level]*j
        box('ceiling',3,0,clearance+.08,1.5,1.3,.08);p.update(clearance=clearance)
    elif family=='narrow_passage':
        width=[.8,.6,.42][level]*j
        for side in [-1,1]:box('wall_'+str(side),3,side*(width/2+.1),.6,1.7,.1,.6)
        p.update(passage_width=width)
    elif family in ['mixed_course','long_course']:
        n=4 if family=='mixed_course' else 12
        order=rng.permutation(n).tolist();amplitude=[.12,.22,.35][level]*j
        for k in range(n):
            h=amplitude*float(rng.uniform(.5,1));half=.1 if order[k]%2 else .6
            box('obstacle_'+str(k),1.7+2.5*k,float(rng.uniform(-.08,.08)),h/2,half,1.5,h/2)
        goal=1.7+2.5*(n-1)+1.5;p.update(obstacle_count=n,order=order,max_height=amplitude)
    elif family!='flat':raise ValueError(family)
    return ET.tostring(root,encoding='unicode'),{'goal_x':goal,'goal_surface_z':top,'spawn_surface_z':start,
        'parameters':p,'primitives':primitives,'task':'command_conditioned_locomotion','image_goal':False}

def overview(meta,path):
    shapes=[];scale=32.;shift=100
    for p in meta['primitives']:
        x=shift+p['x']*scale;y=110-p['y']*scale;w=2*p['sx']*scale;h=2*p['sy']*scale
        shapes.append(f'<rect x="{x-w/2:.1f}" y="{y-h/2:.1f}" width="{w:.1f}" height="{h:.1f}" fill="#98b4ca" stroke="#314b61" stroke-width=".5"/>')
    width=max(420,int(shift+meta['goal_x']*scale+80))
    svg=f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="220" viewBox="0 0 {width} 220"><rect width="100%" height="100%" fill="#f7f9fc"/><text x="12" y="22" font-family="sans-serif" font-size="14">{meta["family"]} / level {meta["level"]+1} / top view (not RGB observation)</text>{"".join(shapes)}<circle cx="100" cy="110" r="5" fill="#198463"/><circle cx="{shift+meta["goal_x"]*scale}" cy="110" r="5" fill="#c55336"/><text x="12" y="206" font-family="sans-serif" font-size="12">Green: start; orange: goal. Heights and gaps are specified in scene JSON.</text></svg>'
    path.write_text(svg)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',default=str(ROOT/'datasets/procedural_v1'));args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    for name in ['scenes','manifests','previews','validation']:(out/name).mkdir()
    cases=[];scene_index=[]
    for split,(base,layouts) in SPLITS.items():
        for fi,family in enumerate(FAMILIES):
            for level in range(3):
                for layout in range(layouts):
                    seed=base+fi*100+level*20+layout
                    sid=f'{split}_{family}_L{level+1}_{layout:02d}'
                    xml,meta=generate(family,level,seed)
                    meta.update(scene_id=sid,family=family,level=level,layout_seed=seed,split=split,
                        xml_sha256=hashlib.sha256(xml.encode()).hexdigest())
                    (out/'scenes'/f'{sid}.xml').write_text(xml)
                    (out/'scenes'/f'{sid}.json').write_text(json.dumps(meta,indent=2))
                    scene_index.append({k:v for k,v in meta.items() if k!='primitives'})
                    if split=='development' and layout==0:overview(meta,out/'previews'/f'{family}_L{level+1}.svg')
                    conds=CONDITIONS if family in ['flat','box','stairs_up'] and level==1 else ['clean']
                    for trial in range(3):
                        eseed=seed*10+trial;r=np.random.default_rng(eseed)
                        basecase={'scene_id':sid,'scene_file':str(out/'scenes'/f'{sid}.xml'),'terrain':family,'height':meta['parameters'].get('height',0.),
                            'difficulty':level+1,'seed':eseed,'yaw':float(r.uniform(-.2,.2)),
                            'x':float(r.uniform(-.12,.12)),'y':float(r.uniform(-.10,.10)),
                            'spawn_z':meta['spawn_surface_z']+.38,'goal_x':meta['goal_x'],'goal_z':meta['goal_surface_z'],
                            'scan_phase':int(r.integers(0,1000)),'speed':.75,
                            'duration':max(20.,meta['goal_x']/.4+8),'split':split,'pair_id':f'{sid}_T{trial}',
                            'protocol':'independent_scenario_v1','image_goal':False}
                        for cond in conds:
                            cat='terrain'
                            if family in ['mixed_course','long_course']:cat='composition' if family=='mixed_course' else 'long_horizon'
                            if cond not in ['clean']:
                                cat='dynamics' if cond in ['friction_low','mass_plus20','push_lateral','speed_slow','speed_fast'] else ('memory_intervention' if cond in ['blind','reset_memory'] else 'perception')
                            cases.append({**basecase,'condition':cond,'category':cat,'episode_id':f'{sid}_T{trial}_{cond}'})
    for split in SPLITS:
        items=[c for c in cases if c['split']==split]
        (out/'manifests'/f'{split}.jsonl').write_text(''.join(json.dumps(c)+'\n' for c in items))
    (out/'scene_index.json').write_text(json.dumps(scene_index,indent=2))
    summary={'version':'procedural_v1','scenes':len(scene_index),'episodes':len(cases),'families':FAMILIES,
             'splits':dict(collections.Counter(c['split'] for c in cases)),
             'categories':dict(collections.Counter(c['category'] for c in cases)),
             'conditions':CONDITIONS,'meaning':'scenario specifications, NOT completed rollout trajectories',
             'official_benchmark':False,'training_performed':False,
             'heldout_scope':'disjoint layout/random seeds within our evaluation; flat geometry shared intentionally; no claim disjoint from unknown pretrained training distribution'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    for name in ['scenes','manifests','previews','validation']:
        (out/name/'README.md').write_text({'scenes':'# 场景\n\nXML 为可加载的 MuJoCo 场景；同名 JSON 记录几何、难度、种子、起点和终点。机器人网格引用服务器的官方资源。\n',
          'manifests':'# 任务清单\n\n每行是一个固定测试任务。development 用于调试，validation 用于确定测试设置，test 不用于挑选模型或调整阈值。同一 pair_id 使用相同场景与初态，只有扰动条件变化。这些是评测划分，不是模型训练数据划分。\n',
          'previews':'# 场景示意图\n\nSVG 为俯视几何示意图，不是机器人观测。模型实际接收三维仿真产生的 LiDAR 深度及本体状态。\n',
          'validation':'# 数据质量检查\n\n保存场景加载、起点碰撞、有限值检查及开发集闭环试运行记录；不代表正式性能测试完成。\n'}[name],encoding='utf-8')

if __name__=='__main__':main()
