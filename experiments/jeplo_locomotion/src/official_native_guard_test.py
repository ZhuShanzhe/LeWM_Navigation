"""Negative launch tests: prove drift/occupied ports reject before starting a simulator."""
import json,socket,subprocess,sys,tempfile
from pathlib import Path
runner=Path(sys.argv[1]);root=Path(sys.argv[2]);checks=[]
with tempfile.TemporaryDirectory(prefix='native_guard_',dir=root) as temp:
    temp=Path(temp)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',5562))
        p=subprocess.run([sys.executable,str(runner),'--root',str(root),'--output',str(temp/'occupied')],capture_output=True,text=True,timeout=15)
        assert p.returncode!=0 and 'Address already in use' in p.stderr,p.stderr
        checks.append({'test':'occupied_loopback_command_port','rejected_before_launch':True})
    fake=temp/'fake';fake.mkdir();(fake/'artifact.bin').write_bytes(b'test-only-fixture')
    (fake/'native_build_manifest.json').write_text(json.dumps({'artifacts':{'artifact.bin':'0'*64}}))
    p=subprocess.run([sys.executable,str(runner),'--root',str(fake),'--output',str(temp/'drift')],capture_output=True,text=True,timeout=15)
    assert p.returncode!=0 and 'Native build artifact drift' in p.stderr,p.stderr
    checks.append({'test':'artifact_hash_drift','rejected_before_launch':True})
report={'pass':True,'tests':checks,'scope':'negative launcher preflight, no simulator or controller launched'}
(root/'native_guard_test.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
