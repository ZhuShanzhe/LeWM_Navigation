"""Illustrative deterministic replays; not additional independent trials."""
import os
os.environ['MUJOCO_GL']='egl'
import json
from pathlib import Path
import deployment_eval_v2 as ev
root=ev.ROOT/'runs/deployment_v2_230';out=root/'videos';out.mkdir(exist_ok=False)
rows=[json.loads(s) for s in (root/'results.jsonl').read_text().splitlines()]
manifest={c['episode_id']:c for c in json.loads((root/'manifest.json').read_text())}
policy=ev.Policy();replays=[]
for group,wanted in [('box_0.3',False),('stairs_0.15',True),('rotate_left',True)]:
    ref=next(r for r in rows if r['group']==group and r['action_mode']=='deployment_default' and r['success']==wanted)
    c=manifest[ref['episode_id']]
    result,_=ev.run(c,policy,False,str(out/(group+'.mp4')))
    replays.append({'episode_id':c['episode_id'],'group':group,'original_reason':ref['reason'],'replay_reason':result['reason'],'position_difference_max':max(abs(a-b) for a,b in zip(ref['position'],result['position']))})
    print(json.dumps(replays[-1]),flush=True)
(out/'replay_checks.json').write_text(json.dumps(replays,indent=2))
import imageio.v2 as imageio
from PIL import Image,ImageDraw
sheet=Image.new('RGB',(960,810),'white');draw=ImageDraw.Draw(sheet)
for row,group in enumerate(['box_0.3','stairs_0.15','rotate_left']):
    reader=imageio.get_reader(str(out/(group+'.mp4')));frames=list(reader.iter_data())
    for col,fraction in enumerate([.15,.5,.9]):
        frame=frames[min(len(frames)-1,int(len(frames)*fraction))]
        sheet.paste(Image.fromarray(frame).resize((320,240)),(320*col,270*row))
        draw.text((320*col+8,270*row+246),f'{group} / {fraction:.0%} of replay',fill='black')
    reader.close()
sheet.save(out/'replay_contact_sheet.jpg',quality=92)
print('videos complete',flush=True)
