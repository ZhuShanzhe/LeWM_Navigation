"""Six-action RPC/interface regression. No model, prompt, navigation score or API call."""
import fcntl
import json
import os
import tempfile
from pathlib import Path
import numpy as np
from paper_freeze import ROOT, CHAIN, DATA, frozen, sha
from paper_eval import RPC
from paper_scope import panel_rows


def main():
    results = []
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        for task in ['place','instance']:
            folder = ROOT/'results/repair_20261019'/('collision_gate_'+task)
            folder.mkdir(parents=True, exist_ok=True)
            episode = panel_rows(task)[0]
            frozen(folder/'panel.jsonl', (json.dumps(episode, sort_keys=True)+'\n').encode())
            with tempfile.TemporaryDirectory(prefix='collision_gate_',dir='/dev/shm') as temp:
                buffer = Path(temp)/'frame.npz'
                rpc = RPC([str(DATA/'env/bin/python'),'-B','-u',str(ROOT/'tools/paper_sim.py'),
                    '--panel',str(folder/'panel.jsonl'),'--output',str(folder),'--buffer',str(buffer),
                    '--max-steps','6'],folder/'simulator_stderr.log',dict(os.environ))
                try:
                    response = rpc.call({'op':'reset'})
                    assert not response['done']
                    for i,action in enumerate([1,2,3,4,5,0]):
                        response = rpc.call({'op':'step','action':action,'aborted':False})
                        assert response['done'] == (i == 5)
                        frame = np.load(buffer)
                        assert frame['rgb'].shape == (480,640,3)
                        assert frame['depth'].shape == (480,640,1)
                        assert np.isfinite(frame['depth']).all() and frame['depth'].min() >= 0
                        assert frame['depth'].max() <= 5./30+1e-6
                finally:
                    rpc.close()
            rows = [json.loads(line) for line in (folder/(episode['episode_id']+'.steps.jsonl')).read_text().splitlines()]
            assert [x['action'] for x in rows] == [1,2,3,4,5,0]
            assert all(isinstance(x['collided'],bool) for x in rows)
            assert all(np.linalg.norm(np.asarray(x['after'])-x['navmesh_expected_after']) < 1e-4 for x in rows)
            results.append(dict(task=task, actions=6, native_forward_collision=rows[0]['collided'],
                trace_sha256=sha(folder/(episode['episode_id']+'.steps.jsonl'))))
    result = dict(passed=True, tasks=results, model_calls=0, formal_scores=False,
        sim_sha256=sha(ROOT/'tools/paper_sim.py'), collision_source='Raw Simulator.step observation collided field')
    frozen(ROOT/'results/repair_20261019/collision_interface_gate.json',result)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
