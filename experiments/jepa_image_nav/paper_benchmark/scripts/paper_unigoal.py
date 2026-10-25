"""Same released UniGoal framework, isolated common-configuration simulator transport."""
import argparse
import contextlib
import fcntl
import functools
import json
import os
import selectors
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

from paper_freeze import ROOT, DATA, CHAIN, frozen, sha
from paper_scope import require_geometry
sys.path.insert(0, str(CHAIN/'tools'))
import chain_unigoal_run as native


class CommonTransport(native.SensorTransport):
    def __init__(self, panel, out, category_assisted, max_steps):
        assert not category_assisted
        self.temporary = tempfile.TemporaryDirectory(prefix='paper_uni_', dir='/dev/shm')
        self.buffer = Path(self.temporary.name)/'sensors.npz'
        self.log = (out/'simulator_stderr.log').open('a')
        self.process = subprocess.Popen([str(DATA/'env/bin/python'), str(ROOT/'tools/paper_sim.py'),
            '--panel', str(panel), '--output', str(out), '--buffer', str(self.buffer), '--max-steps', str(max_steps)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log, text=True, bufsize=1,
            env=dict(os.environ, MAGNUM_LOG='quiet', HABITAT_SIM_LOG='quiet', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1'))
        self.poll = selectors.DefaultSelector()
        self.poll.register(self.process.stdout, selectors.EVENT_READ)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--panel', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-actions', type=int, default=500)
    args = parser.parse_args()
    if args.max_actions == 500:
        from paper_runtime import verify
        verify()
    require_geometry()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    original_load = native.yaml.safe_load
    def configured(stream):
        value = original_load(stream)
        # Native execute() sets goal_type after loading this base YAML.
        assert isinstance(value, dict)
        assert all(k in value for k in ['env_frame_width', 'env_frame_height', 'map_size_cm', 'map_resolution'])
        value.update(env_frame_width=640, env_frame_height=480, frame_width=160, frame_height=120,
            camera_height=.88, hfov=79, turn_angle=30, min_depth=0., max_depth=30.)
        return value
    native.yaml = SimpleNamespace(safe_load=configured)
    native.SensorTransport = CommonTransport
    frozen(out/'common_configuration.json', {'model': 'unigoal', 'task': 'instance',
        'contract_sha256': sha(ROOT/'protocol/common_v1.json'), 'panel_sha256': sha(Path(args.panel)),
        'depth_transport': 'Physical clip0..5m, native normalized /30, graph converts *30; BEV retains native convention',
        'source_sha256': sha(CHAIN/'tools/chain_unigoal_run.py'),
        'native_changes': 'Only declared common camera/turn/depth range and simulator/scoring transport; native graph/BEV/FMM/models preserved'})
    with (CHAIN/'gpu_benchmark.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        from chain_gpu_idle import wait
        wait()
        # Reuse only the local HTTP transport within this cold-start episode.
        # Weights, inference calls, RNG, prompts, retries and episode state stay native.
        gate_path = ROOT/'results/unigoal_client_pool_gate.json'
        client_gate = json.loads(gate_path.read_text())
        assert client_gate['passed'] and client_gate['adapter_sha256'] == sha(ROOT/'tools/paper_unigoal_client_pool.py')
        from paper_unigoal_client_pool import pooled_clients
        original_prepare = native.prepare
        stage_times = {}; original_methods = []; pool_reports = []
        def instrument(cls, name, label):
            if not hasattr(cls, name):
                return
            original = getattr(cls, name); original_methods.append((cls, name, original))
            @functools.wraps(original)
            def measured(*a, **kw):
                started = time.monotonic()
                try:
                    return original(*a, **kw)
                finally:
                    item = stage_times.setdefault(label, {'calls': 0, 'seconds': 0.})
                    item['calls'] += 1; item['seconds'] += time.monotonic()-started
            setattr(cls, name, measured)
        started = time.monotonic()
        with contextlib.ExitStack() as stack:
            diagnostic_gate = json.loads((ROOT/'results/unigoal_call_diagnostics_gate.json').read_text())
            assert diagnostic_gate['passed'] and diagnostic_gate['adapter_sha256'] == sha(ROOT/'tools/paper_unigoal_call_diagnostics.py')
            from paper_unigoal_call_diagnostics import call_diagnostics
            stack.enter_context(call_diagnostics(out))
            load_gate_path = ROOT/'results/unigoal_actual_model_load_gate.json'
            load_gate = json.loads(load_gate_path.read_text())
            assert load_gate['passed'] and load_gate['bit_exact_model_states_and_fixed_image_outputs']
            load_reports = []
            if load_gate['activate_mmap']:
                assert load_gate['adapter_sha256'] == sha(ROOT/'tools/paper_unigoal_mmap.py')
                from paper_unigoal_mmap import mapped_checkpoints
                load_reports.append(stack.enter_context(mapped_checkpoints(native.torch)))
            def prepared():
                value = original_prepare()
                import src.utils.llm as local_llm
                pool_reports.append(stack.enter_context(pooled_clients(local_llm)))
                for cls, names, prefix in [(value[0], ['__init__', 'set_image_goal', 'update_scenegraph', 'explore'], 'graph'),
                    (value[1], ['__init__', 'reset', 'step'], 'agent'), (value[2], ['__init__', 'mapping'], 'BEV')]:
                    for name in names:
                        instrument(cls, name, prefix+'.'+name)
                return value
            native.prepare = prepared
            try:
                native.execute(SimpleNamespace(run=str(out), panel=args.panel, limit=0,
                    category_assisted=False, max_steps=args.max_actions, sim_gate=False,
                    base_url='http://127.0.0.1:11435/v1/'))
            finally:
                native.prepare = original_prepare
                for cls, name, original in reversed(original_methods):
                    setattr(cls, name, original)
                frozen(out/'timing_profile.json', {'policy_process_execute_seconds': time.monotonic()-started,
                    'stages': stage_times, 'stage_times_are_inclusive_and_must_not_be_summed': True,
                    'cuda_synchronized_for_timing': False, 'http_client_pool_gate_sha256': sha(gate_path),
                    'client_pool_statistics': pool_reports,
                    'checkpoint_loading_statistics': load_reports,
                    'mmap_activated_after_model_gate': load_gate['activate_mmap'],
                    'actual_model_load_gate_sha256': sha(load_gate_path),
                    'client_pooling': 'Same local client reused inside one episode; all clients closed at episode end',
                    'cold_start_episode_isolation_preserved': True})
        subprocess.run([str(CHAIN/'goat_env/bin/python'), str(ROOT/'tools/paper_accept.py'),
            '--episode-folder', str(out), '--configuration', str(out/'common_configuration.json'),
            '--max-actions', str(args.max_actions)], check=True)


if __name__ == '__main__':
    main()
