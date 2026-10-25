"""Common physical image-goal evaluation using unchanged released policy services."""
import argparse
import fcntl
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from paper_freeze import ROOT, DATA, CHAIN, frozen, sha
from paper_common import CONTRACT
from paper_scope import panel_path, panel_rows, run_name, require_geometry


class RPC:
    def __init__(self, command, log, env):
        self.stream = log.open('a')
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self.stream, text=True, bufsize=1, env=env)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def call(self, request=None, timeout=180):
        if request is not None:
            self.process.stdin.write(json.dumps(request)+'\n'); self.process.stdin.flush()
        assert self.selector.select(timeout), 'Native service timeout'
        line = self.process.stdout.readline(); assert line, 'Native service exited'
        response = json.loads(line); assert 'error' not in response, response
        return response

    def close(self):
        if self.process.poll() is None:
            try:
                self.call({'op': 'close'}, timeout=1)
            except Exception:
                pass
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate(); self.process.wait(timeout=15)
        self.selector.close(); self.stream.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['fgprompt', 'debit', 'goat'], required=True)
    parser.add_argument('--task', choices=['place', 'instance'], required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not args.smoke:
        from paper_runtime import verify
        verify()
    assert args.model == 'goat' or args.task == 'place', 'No place-model transfer as native instance baseline'
    panel = panel_path(args.task)
    rows = panel_rows(args.task)
    if args.smoke:
        rows = [rows[0]]
    out = ROOT/'results'/run_name(args.model, args.task, args.smoke)
    out.mkdir(parents=True, exist_ok=True)
    require_geometry()
    env = dict(os.environ, NO_PROXY='localhost,127.0.0.1', no_proxy='localhost,127.0.0.1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LD_PRELOAD='/usr/lib/x86_64-linux-gnu/libstdc++.so.6')
    old = Path('/root/autodl-tmp/jepa_vln_baselines_20261015')
    command = ([str(CHAIN/'goat_env/bin/python'), str(CHAIN/'tools/instance_policy_goat.py')]
        if args.model == 'goat' else [str(old/'policy_env/bin/python'), str(old/'tools/baseline_policy.py'), '--model', args.model])
    with (out/'batch.lock').open('a') as batch, (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(batch, fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(lock, fcntl.LOCK_EX)
        sys.path.insert(0, str(CHAIN/'tools')); from chain_gpu_idle import wait
        wait()
        policy = RPC(command, out/'policy_stderr.log', env)
        try:
            meta = policy.call(); assert meta['ready']
            frozen(out/'configuration.json', {'model': args.model, 'task': args.task,
                'panel_sha256': sha(panel), 'contract_sha256': sha(ROOT/'protocol/common_v1.json'),
                'episodes': len(rows), 'policy': meta, 'modalities': CONTRACT['policy_inputs'][{'goat': 'GOAT', 'fgprompt': 'FGPrompt', 'debit': 'DEBiT'}[args.model]],
                'native_source': str(command[1]), 'native_source_sha256': sha(Path(command[1]))})
            for e in rows:
                folder = out/'episodes'/e['episode_id']; folder.mkdir(parents=True, exist_ok=True)
                if (folder/'accepted_summary.json').exists():
                    continue
                from paper_archive import archive
                archive(folder, ['runtime_failure.json', 'records.json', e['episode_id']+'.json',
                    e['episode_id']+'.steps.jsonl', 'policy_actions.jsonl', 'simulator_stderr.log', 'progress.json'])
                query = folder/'panel.jsonl'
                frozen(query, (json.dumps(e, sort_keys=True)+'\n').encode())
                sim = None
                try:
                    with tempfile.TemporaryDirectory(prefix='paper_eval_', dir='/dev/shm') as temporary:
                        temp = Path(temporary); buffer = temp/'obs.npz'; rgb = temp/'rgb.npy'; goal = temp/'goal.npy'
                        sim = RPC([str(DATA/'env/bin/python'), '-u', str(ROOT/'tools/paper_sim.py'),
                            '--panel', str(query), '--output', str(folder), '--buffer', str(buffer),
                            '--max-steps', '20' if args.smoke else '500'], folder/'simulator_stderr.log', env)
                        reply = sim.call({'op': 'reset'})
                        with np.load(buffer) as observations:
                            np.save(goal, observations['goal_rgb'])
                        policy.call({'op': 'reset', 'goal': str(goal)})
                        with (folder/'policy_actions.jsonl').open('x') as actions:
                            while not reply['done']:
                                with np.load(buffer) as observations:
                                    np.save(rgb, observations['rgb'])
                                    request = {'op': 'step', 'rgb': str(rgb)}
                                    if args.model == 'goat':
                                        request.update(gps=observations['gps'].tolist(), compass=float(observations['compass'][0]))
                                decision = policy.call(request)
                                native_action = int(decision['action'])
                                aborted = args.model == 'goat' and native_action == 0
                                action = 0 if args.model == 'goat' and native_action == 6 else native_action
                                actions.write(json.dumps({'native_action': native_action, 'canonical_action': action,
                                    'aborted': aborted, **{k: v for k, v in decision.items() if k != 'action'}})+'\n'); actions.flush()
                                reply = sim.call({'op': 'step', 'action': action, 'aborted': aborted})
                        sim.close(); sim = None
                    subprocess.run([sys.executable, str(ROOT/'tools/paper_accept.py'), '--episode-folder', str(folder),
                        '--configuration', str(out/'configuration.json'), '--max-actions', '20' if args.smoke else '500'], check=True)
                    print(json.dumps({'completed': e['episode_id'], 'run': out.name}), flush=True)
                except Exception as exc:
                    (folder/'runtime_failure.json').write_text(json.dumps({'error': type(exc).__name__,
                        'detail': str(exc), 'unfinished_counted_as_failure': False}))
                    raise
                finally:
                    if sim is not None:
                        sim.close()
        finally:
            policy.close()


if __name__ == '__main__':
    main()
