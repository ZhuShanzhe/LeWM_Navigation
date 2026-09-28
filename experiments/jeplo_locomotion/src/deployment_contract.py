"""Offline-tested navigation/deployment boundary. Does not send robot commands.

Freshness gating must be called continuously by an integration loop. This helper
does not repair the upstream C++ receiver's missing watchdog, nor certify safety.
"""
import json,math,struct,unittest
from pathlib import Path
import numpy as np

def body_command(world_vx,world_vy,yaw,wz):
    a=np.asarray([world_vx,world_vy,yaw,wz],dtype=float)
    if not np.isfinite(a).all():raise ValueError('Nonfinite command/pose')
    c,s=math.cos(yaw),math.sin(yaw)
    return np.array([c*world_vx+s*world_vy,-s*world_vx+c*world_vy,wz])

def pack_command(cmd):
    a=np.asarray(cmd,dtype=float)
    if a.shape!=(3,) or not np.isfinite(a).all():raise ValueError('Expected finite vx,vy,wz')
    a=np.clip(a,[-1.5,-1.5,-3],[1.5,1.5,3])
    return struct.pack('<3f',*a)

def pack_depth(depth):
    a=np.asarray(depth,dtype=np.float32)
    if a.shape!=(25,60) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):
        raise ValueError('Expected normalized finite 25x60 range image')
    return struct.pack('<II',60,25)+a.astype('<f4').tobytes(order='C')

def gated_command(cmd,now,command_time,depth_time,timeout=.25):
    if not np.isfinite([now,command_time,depth_time,timeout]).all() or timeout<=0:
        raise ValueError('Invalid time')
    if any(now-t<0 or now-t>timeout for t in [command_time,depth_time]):
        return pack_command([0,0,0])
    return pack_command(cmd)

class ContractTests(unittest.TestCase):
    def test_command_bytes(self):
        p=pack_command([.75,-.5,1]);self.assertEqual(len(p),12)
        self.assertEqual(struct.unpack('<3f',p),(.75,-.5,1))
    def test_limits(self):self.assertEqual(struct.unpack('<3f',pack_command([9,-9,9])),(1.5,-1.5,3))
    def test_bad_command(self):
        for c in [[1,2],[1,float('nan'),0],[0,0,float('inf')]]:
            with self.assertRaises(ValueError):pack_command(c)
    def test_rotation(self):np.testing.assert_allclose(body_command(1,0,math.pi/2,.5),[0,-1,.5],atol=1e-7)
    def test_depth(self):
        d=np.arange(1500,dtype=np.float32).reshape(25,60)/1500;p=pack_depth(d)
        self.assertEqual(len(p),6008);self.assertEqual(struct.unpack('<II',p[:8]),(60,25))
        np.testing.assert_equal(np.frombuffer(p[8:],dtype='<f4').reshape(25,60),d)
    def test_bad_depth(self):
        for a in [np.ones((60,25)),np.full((25,60),np.nan),np.full((25,60),2)]:
            with self.assertRaises(ValueError):pack_depth(a)
    def test_command_timeout(self):self.assertEqual(gated_command([1,0,0],1,.7,1),pack_command([0,0,0]))
    def test_depth_timeout(self):self.assertEqual(gated_command([1,0,0],1,1,.7),pack_command([0,0,0]))
    def test_future_timestamp(self):self.assertEqual(gated_command([1,0,0],1,2,1),pack_command([0,0,0]))
    def test_fresh(self):self.assertEqual(gated_command([1,0,0],1,.9,.9),pack_command([1,0,0]))

if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ContractTests))
    import os
    dest=Path(os.environ.get('JEPLO_WORKSPACE','/root/autodl-tmp/jeplo_eval_20260928'))/'runs/deployment_v2_230/contract_tests.json'
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps({'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'pass':result.wasSuccessful(),'scope':'offline serialization and helper watchdog only; no live ZMQ/DDS/robot'},indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)
