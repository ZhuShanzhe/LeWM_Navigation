"""Exercise official C++ ZMQ projection asynchronously with synthetic point packets.
No robot controller, no physical-performance claims. Separate clean/cage conditions.
"""
import argparse,json,struct,subprocess,time
from pathlib import Path
import numpy as np
import zmq

p=argparse.ArgumentParser();p.add_argument('--binary',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
results=[]
for cage in [False,True]:
    ctx=zmq.Context();pub=ctx.socket(zmq.PUB);sub=ctx.socket(zmq.SUB)
    pub.setsockopt(zmq.LINGER,0);sub.setsockopt(zmq.LINGER,0);sub.setsockopt(zmq.SUBSCRIBE,b'')
    inport=pub.bind_to_random_port('tcp://127.0.0.1')
    # Let an OS-selected temporary listener choose an unused output port.
    import socket
    with socket.socket() as reserved:
        reserved.bind(('127.0.0.1',0));outport=reserved.getsockname()[1]
    sub.connect(f'tcp://127.0.0.1:{outport}')
    args=[str(a.binary.resolve()),'--sim','--downsample-rate','4','--fov','25x60','--sim-port',str(inport),'--port',str(outport)]
    if not cage:args+=['--no-cage-mask']
    logpath=a.output.with_name('sensor_cage.log' if cage else 'sensor_clean.log')
    pose=np.r_[np.zeros(3),np.eye(3).ravel()].astype('<f4')
    # All bins receive unit-range points, repeated four times so stride 4 retains them.
    theta=np.deg2rad(-4.5+2*np.arange(25))[:,None];phi=np.deg2rad(-60+2*np.arange(60))[None,:]
    pts=np.stack(np.broadcast_arrays(np.cos(theta)*np.cos(phi),np.cos(theta)*np.sin(phi),np.sin(theta)),axis=-1).reshape(-1,3).astype('<f4')
    packet=pose.tobytes()+np.repeat(pts,4,axis=0).tobytes()
    times=[];frames=[];sent=0
    with logpath.open('w') as log:
        proc=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        try:
            start=time.monotonic();next_send=start
            while time.monotonic()-start<7:
                now=time.monotonic()
                if now>=next_send:
                    pub.send(packet);sent+=1;next_send+=.05
                if sub.poll(timeout=2):
                    raw=sub.recv();stamp=time.monotonic()
                    if stamp-start>2:
                        assert len(raw)==6008 and struct.unpack('<II',raw[:8])==(60,25)
                        frame=np.frombuffer(raw[8:],dtype='<f4').copy()
                        assert np.isfinite(frame).all() and np.all((frame>=0)&(frame<=1))
                        times.append(stamp);frames.append(frame)
                if proc.poll() is not None:raise RuntimeError(f'publisher exited {proc.returncode}')
        finally:
            proc.terminate()
            # Upstream receiver can block in zmq_recv; send a final packet to unblock.
            for _ in range(3):pub.send(packet);time.sleep(.02)
            try:proc.wait(timeout=2)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
            pub.close();sub.close();ctx.term()
    hz=(len(times)-1)/(times[-1]-times[0]) if len(times)>1 else 0
    assert len(frames)>=35 and 8<=hz<=12
    last=frames[-1];observed=np.isclose(last,.5,atol=1e-6);missing=last==1
    assert np.all(observed|missing)
    if not cage:assert observed.all()
    else:assert observed.any() and missing.any()
    results.append({'cage_mask':cage,'sent_20hz_packets':sent,'received_after_warmup':len(times),'measured_publish_hz':hz,'max_period_ms':1000*max(np.diff(times)),'normalized_half_range_bins':int(observed.sum()),'masked_bins':int(missing.sum()),'observed_mask_flat_indices':np.flatnonzero(missing).tolist(),'pass':True})
report={'scope':'official C++ sensor ZMQ path, synthetic packets only; no simulator/controller/robot','cases':results,'pass':all(x['pass'] for x in results)}
a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
