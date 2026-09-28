"""Frozen official fixtures on pinned ORT/TensorRT. Not a locomotion benchmark."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import onnxruntime as ort
p=argparse.ArgumentParser();p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
assert ort.__version__=='1.22.0',ort.__version__
assert 'TensorrtExecutionProvider' in ort.get_available_providers()
report={'scope':'official ONNX samples, ORT Python API with TensorRT provider; not the native controller or closed-loop','onnxruntime':ort.__version__,'python':platform.python_version(),'models':{}}
for name in ['policy','sensor_estimator']:
    opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.enable_profiling=True
    opts.profile_file_prefix=str(a.output.parent/('ort_profile_'+name))
    start=time.monotonic()
    sess=ort.InferenceSession(str(a.model_dir/(name+'.onnx')),opts,providers=[('TensorrtExecutionProvider',{'trt_fp16_enable':False,'trt_engine_cache_enable':False}), 'CPUExecutionProvider'])
    assert 'TensorrtExecutionProvider' in sess.get_providers(),'TensorRT silently fell back'
    errors=[]
    for f in sorted((a.model_dir/'test_cases'/name).glob('*.npz')):
        with np.load(f) as data:
            feeds={s.name:data['input__'+s.name] if 'input__'+s.name in data else data[s.name] for s in sess.get_inputs()}
            outputs=sess.run(None,feeds)
            for spec,pred in zip(sess.get_outputs(),outputs):
                ref=data['output__'+spec.name] if 'output__'+spec.name in data else data[spec.name]
                delta=np.abs(pred-ref)
                errors.append({'sample':f.name,'output':spec.name,'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'finite':bool(np.isfinite(pred).all())})
    assert errors
    profile=Path(sess.end_profiling());events=json.loads(profile.read_text())
    executed=sorted({x.get('args',{}).get('provider') for x in events if x.get('args',{}).get('provider')})
    # Advertised providers alone are insufficient: require evidence of actual TRT kernels.
    trt_executed='TensorrtExecutionProvider' in executed
    tol=.1 if name=='policy' else .5
    passed=trt_executed and all(e['finite'] and e['max_abs']<=tol and e['mean_abs']<=.1 for e in errors)
    report['models'][name]={'providers_executed':executed,'providers_registered':sess.get_providers(),'errors':errors,'official_tolerance_pass':passed,'elapsed_seconds':time.monotonic()-start,'sha256':hashlib.file_digest((a.model_dir/(name+'.onnx')).open('rb'),'sha256').hexdigest()}
    a.output.write_text(json.dumps(report,indent=2))
    assert passed,report['models'][name]
report['pass']=True;a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
