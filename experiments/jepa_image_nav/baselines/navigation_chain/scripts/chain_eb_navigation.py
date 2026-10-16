"""EB-Navigation native tasks/planner; local Qwen transport, not a new planner.

Private frames/responses remain on the server. No dataset labels, positions or
target identifiers are provided to the model. Original action feedback is kept.
"""
import argparse
import base64
import copy
import fcntl
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np
from PIL import Image

ROOT = Path('/root/autodl-tmp/jepa_instance_bench_20261016')
REPO = ROOT / 'repos/EmbodiedBench'
OUT = ROOT / 'results/eb_navigation_qwen25vl7b'
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.WARNING)
# The publisher entry module imports all unrelated simulators/providers. Only
# its logger is used by the actual native navigation modules imported below.
entry = types.ModuleType('embodiedbench.main')
entry.logger = logging.getLogger('eb_navigation')
sys.modules[entry.__name__] = entry

def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)

class LocalQwenTransport:
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls)
        return cls.instance

    def __init__(self, *args, **kwargs):
        if hasattr(self, 'model'):
            return
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        import xgrammar as xgr
        import xgrammar.contrib.hf
        from embodiedbench.planner.planner_config.generation_guide import vlm_generation_guide
        self.torch = torch
        torch.set_num_threads(2)
        torch.manual_seed(20261016)
        folder = ROOT / 'models/Qwen2.5-VL-7B-Instruct'
        self.processor = AutoProcessor.from_pretrained(folder, trust_remote_code=False, local_files_only=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(folder,
            torch_dtype=torch.bfloat16, device_map={'': 'cuda:0'},
            attn_implementation='sdpa', trust_remote_code=False, local_files_only=True).eval()
        self.xgrammar = xgr
        token_info = xgr.TokenizerInfo.from_huggingface(self.processor.tokenizer,
            vocab_size=self.model.config.vocab_size)
        self.grammar = xgr.GrammarCompiler(token_info).compile_json_schema(json.dumps(vlm_generation_guide))
        self.responses = []

    def respond(self, messages):
        self.last_transport_error = None
        try:
            return self._respond(messages)
        except Exception as exc:
            self.last_transport_error = type(exc).__name__ + ': ' + str(exc)
            raise

    def _respond(self, messages):
        converted, images = [], []
        for message in copy.deepcopy(messages):
            content = []
            for piece in message['content']:
                if piece['type'] == 'image_url':
                    uri = piece['image_url']['url']
                    assert uri.startswith('data:image/') and ';base64,' in uri
                    im = Image.open(io.BytesIO(base64.b64decode(uri.split(',', 1)[1]))).convert('RGB')
                    content.append({'type': 'image', 'image': im})
                    images.append(im)
                else:
                    assert piece['type'] == 'text'
                    content.append(piece)
            converted.append({'role': message['role'], 'content': content})
        prompt = self.processor.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, padding=True, return_tensors='pt').to('cuda')
        self.torch.cuda.synchronize()
        start = time.perf_counter()
        with self.torch.inference_mode():
            tokens = self.model.generate(**inputs, max_new_tokens=4096, do_sample=False,
                logits_processor=[self.xgrammar.contrib.hf.LogitsProcessor(self.grammar)])
        self.torch.cuda.synchronize()
        result = self.processor.batch_decode(tokens[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        record = {'response': result, 'seconds': time.perf_counter()-start,
            'input_tokens': int(inputs.input_ids.shape[1]),
            'output_tokens': int(tokens.shape[1]-inputs.input_ids.shape[1]),
            'image_count': len(images)}
        self.responses.append(record)
        with (OUT/'model_calls.private.jsonl').open('a') as log:
            log.write(json.dumps(record)+'\n')
        return result

transport = types.ModuleType('embodiedbench.planner.remote_model')
transport.RemoteModel = LocalQwenTransport
sys.modules[transport.__name__] = transport

def execute(args):
    import ai2thor.build
    import ai2thor.controller
    # Same publisher asset/hash, HTTPS transport instead of obsolete HTTP URL.
    ai2thor.build.base_url = 'https://ai2-thor-public.s3.amazonaws.com/'
    assert (ROOT/'results/ai2thor_runtime.json').exists(), 'Prepare native binary without holding the GPU lock first'
    NativeController = ai2thor.controller.Controller
    class ProjectCacheController(NativeController):
        @property
        def base_dir(self):
            return str(ROOT/'runtime/ai2thor')
    ai2thor.controller.Controller = ProjectCacheController
    from embodiedbench.envs.eb_navigation.EBNavEnv import EBNavigationEnv, ValidEvalSets
    from embodiedbench.planner.nav_planner import EBNavigationPlanner
    from embodiedbench.evaluator.config.system_prompts import eb_navigation_system_prompt
    from embodiedbench.evaluator.config.eb_navigation_example import examples
    import yaml
    native = yaml.safe_load((REPO / 'embodiedbench/configs/eb-nav.yaml').read_text())
    assert native['n_shots'] == 3 and not native['chat_history']
    assert native['resolution'] == 500 and not native['detection_box']
    OUT.mkdir(parents=True, exist_ok=True)
    config = dict(native, model_name='Qwen/Qwen2.5-VL-7B-Instruct',
        model_transport='local Transformers 4.51.3, BF16, SDPA, official JSON schema via XGrammar 0.2.8',
        max_new_tokens=4096, temperature=0, simulation='AI2THOR 5.0 CloudRendering',
        source_commit='9be4e980e9cd6bcb38373cd4aab7c32724bdd401',
        evaluation_scope='Official EB-Navigation tasks/actions/prompts/feedback/success; not R2R/RxR scores',
        runtime_failures='Recorded separately, never silently converted to navigation failure')
    save(OUT / 'configuration.json', config)
    os.chdir(OUT)
    sets = [args.subset] if args.subset else ValidEvalSets
    for subset in sets:
        env = EBNavigationEnv(eval_set=subset, exp_name=subset, resolution=500)
        try:
            if args.sim_smoke:
                obs = env.reset()
                event = env.env.last_event
                assert event.metadata['lastActionSuccess'], 'Native start teleport failed'
                assert obs['head_rgb'].shape == (500, 500, 3)
                before = copy.deepcopy(event.metadata['agent'])
                obs, reward, done, info = env.step(4, {}, 1)
                after = env.env.last_event.metadata['agent']
                yaw_delta = (after['rotation']['y']-before['rotation']['y']) % 360
                assert abs(yaw_delta-90) < 1e-3 and info['last_action_success']
                save(OUT / 'simulator_gate.json', {'passed': True, 'native_teleport': True,
                    'rgb_shape': list(obs['head_rgb'].shape), 'right_turn_degrees': yaw_delta,
                    'max_steps': env._max_episode_steps, 'native_success_threshold_m': 1,
                    'scene': env.episode_data['scene'], 'subset': subset})
                print('EB_SIMULATOR_GATE_PASSED', flush=True)
                return
            planner = EBNavigationPlanner(model_name='qwen2.5-vl-7b-instruct', model_type='local',
                actions=env.language_skill_set, system_prompt=eb_navigation_system_prompt,
                examples=examples, n_shot=native['n_shots'], chat_history=native['chat_history'],
                truncate=native['truncate'], multiview=False, multistep=False, visual_icl=False)
            if args.transport_gate:
                obs=env.reset()
                image=env.save_image(obs)
                planner.reset()
                action,reasoning=planner.act(image,env.episode_language_instruction)
                if planner.model.last_transport_error:
                    raise RuntimeError(planner.model.last_transport_error)
                result=json.loads(reasoning)
                assert isinstance(action,list) and action and all(isinstance(a,int) and 0<=a<8 for a in action)
                assert planner.output_json_error==0 and 'executable_plan' in result
                save(OUT/'transport_gate.json',{'passed':True,'native_prompt':True,'native_schema':True,
                    'valid_action_ids':action,'response':result,'generation':planner.model.responses[-1]})
                print('EB_QWEN_TRANSPORT_GATE_PASSED',flush=True)
                return
            count = min(env.number_of_episodes, args.limit or env.number_of_episodes)
            for index in range(count):
                dest = OUT / subset / f'{index:03d}.json'
                if dest.exists() and json.loads(dest.read_text()).get('status') == 'complete':
                    env._current_episode_num += 1
                    continue
                trace = []
                first_response = len(planner.model.responses)
                record = {'subset': subset, 'index': index, 'status': 'runtime_failure'}
                try:
                    np.random.seed(20261016+index)
                    obs = env.reset()
                    assert env.env.last_event.metadata['lastActionSuccess'], 'Native start teleport failed'
                    img = env.save_image(obs)
                    planner.reset()
                    done = False
                    while not done:
                        action, reasoning = planner.act(img, env.episode_language_instruction)
                        if planner.model.last_transport_error:
                            raise RuntimeError(planner.model.last_transport_error)
                        reasoning = json.loads(reasoning)
                        fallback = 'invalid json, random action' in reasoning.get('visual_state_description', '')
                        actions = action if isinstance(action, list) else [action]
                        for k, one in enumerate(actions[:env._max_episode_steps-env._current_step]):
                            before = copy.deepcopy(env.env.last_event.metadata['agent'])
                            obs, reward, done, info = env.step(one, reasoning, int(k == 0))
                            planner.update_info(info)
                            img = env.save_image(obs)
                            trace.append({'t': info['env_step'], 'requested_action': one,
                                'executed_action': info['action_id'], 'fallback': fallback,
                                'before': before, 'after': copy.deepcopy(env.env.last_event.metadata['agent']),
                                'distance': info['distance'], 'task_success': info['task_success'],
                                'action_success': info['last_action_success']})
                            if done or not info['last_action_success']:
                                break
                        if not actions:
                            raise RuntimeError('Planner returned no actions')
                    record.update(status='complete', instruction=env.episode_language_instruction,
                        success=info['task_success'], final_distance=info['distance'], steps=info['env_step'],
                        planner_steps=planner.planner_steps, parser_errors=planner.output_json_error,
                        fallback_steps=sum(s['fallback'] for s in trace),
                        invalid_action_steps=sum(not s['action_success'] for s in trace))
                except Exception as exc:
                    record.update(error=type(exc).__name__, detail=str(exc))
                    traceback.print_exc()
                record['trace'] = trace
                record['responses'] = planner.model.responses[first_response:]
                save(dest, record)
                print(json.dumps({k:v for k,v in record.items() if k not in {'trace','responses','instruction'}}), flush=True)
                if record['status'] != 'complete':
                    raise RuntimeError('Runtime gate failed; stop rather than contaminate full benchmark')
        finally:
            env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', choices=['base','common_sense','complex_instruction','visual_appearance','long_horizon'])
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--sim-smoke', action='store_true')
    parser.add_argument('--transport-gate', action='store_true')
    args = parser.parse_args()
    if args.sim_smoke:
        execute(args)
    else:
        with (ROOT / 'gpu_benchmark.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            execute(args)
