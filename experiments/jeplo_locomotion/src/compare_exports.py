"""Compare the two published export locations without assuming byte equality."""
import json,hashlib
from pathlib import Path
import numpy as np
import onnxruntime as ort
from deployment_eval_v2 import ROOT,REPO,MODEL_DIR
out={}
for name in ['policy','sensor_estimator']:
    a=MODEL_DIR/(name+'.onnx');b=REPO/'training/logs/rsl_rl/go2_loco/pretrained/exported'/(name+'.onnx')
    opt=ort.SessionOptions();opt.intra_op_num_threads=1
    sa=ort.InferenceSession(str(a),opt,providers=['CPUExecutionProvider']);sb=ort.InferenceSession(str(b),opt,providers=['CPUExecutionProvider'])
    item={'deployment_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'training_sha256':hashlib.sha256(b.read_bytes()).hexdigest(),'training_inputs':{x.name:x.shape for x in sb.get_inputs()},'samples':[]}
    for f in sorted((MODEL_DIR/'test_cases'/name).glob('*.npz')):
        z=np.load(f);feeds={s.name:z[s.name] for s in sa.get_inputs()}
        pa=sa.run(None,feeds);pb=sb.run(None,feeds)
        item['samples'].append({'sample':f.name,'max_difference':max(float(np.abs(x-y).max()) for x,y in zip(pa,pb))})
    out[name]=item
(ROOT/'runs/deployment_v2_230/export_comparison.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
