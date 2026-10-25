"""CPU-only nativeLLM/VLM transport regression; never invokes a paid or local model."""
import hashlib
import argparse
import importlib.util
import json
import socket
import threading
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image
import openai
from paper_freeze import ROOT, CHAIN, frozen, sha
from paper_unigoal_client_pool import pooled_clients


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--diagnostics', action='store_true')
    args = parser.parse_args()
    source = CHAIN/'repos/UniGoal/src/utils/llm.py'
    spec = importlib.util.spec_from_file_location('unigoal_client_gate_native_llm', source)
    native = importlib.util.module_from_spec(spec); spec.loader.exec_module(native)
    captured = []
    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'
        def setup(self):
            super().setup(); self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        def log_message(self, *args):
            pass
        def do_POST(self):
            body = self.rfile.read(int(self.headers['Content-Length']))
            captured.append((self.path, body, self.headers.get('Authorization'), self.client_address))
            payload = json.loads(body)
            response = {'id': 'local-transport-fixture', 'object': 'chat.completion', 'created': 0,
                'model': payload['model'], 'choices': [{'index': 0, 'finish_reason': 'stop',
                    'message': {'role': 'assistant', 'content': '0.5'}}],
                'usage': {'prompt_tokens': 5, 'completion_tokens': 2, 'total_tokens': 7}}
            data = json.dumps(response).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers()
            self.wfile.write(data); self.wfile.flush()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler); server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    base_url = f'http://127.0.0.1:{server.server_port}/v1/'
    image = Image.new('RGB', (48, 64), (37, 71, 113))
    def run():
        llm = native.LLM(base_url, 'local-fixture-not-a-secret', 'llama3.2-vision:11b')
        vlm = native.VLM(base_url, 'local-fixture-not-a-secret', 'llama3.2-vision:11b')
        outputs = []; start = time.monotonic()
        for index in range(30):
            prompt = f'Fixture {index}: return probability.'
            outputs.append(vlm(prompt, image) if index % 7 == 0 else llm(prompt))
        return outputs, time.monotonic()-start
    try:
        outputs0, seconds0 = run()
        before = list(captured); captured.clear()
        with pooled_clients(native) as pooling:
            outputs1, seconds1 = run()
        after = list(captured)
        assert outputs0 == outputs1 and len(before) == len(after) == 30
        # Compare serialized JSON bytes including imagePNG, model, full messages,
        # omitted generation fields, method path and authorization. TCPport differs.
        assert [r[:3] for r in before] == [r[:3] for r in after]
        assert len({r[3] for r in before}) == 30
        assert len({r[3] for r in after}) == 1
        assert pooling['factory_calls'] == 30 and pooling['clients_constructed'] == 1
        assert native.OpenAI is openai.OpenAI
        if args.diagnostics:
            from paper_unigoal_call_diagnostics import call_diagnostics
            captured.clear()
            with tempfile.TemporaryDirectory(prefix='unigoal_call_gate_') as temporary:
                folder = Path(temporary)
                with call_diagnostics(folder), pooled_clients(native):
                    outputs2, _ = run()
                telemetry = [json.loads(line) for line in (folder/'model_call_diagnostics.jsonl').read_text().splitlines()]
            assert outputs2 == outputs0 and [r[:3] for r in captured] == [r[:3] for r in before]
            assert len(telemetry) == 30 and all(r['finish_reasons'] == ['stop'] for r in telemetry)
            assert all(r['status'] == 'complete' and r['completion_tokens'] == 2 and r['seconds'] >= 0 for r in telemetry)
            result = {'passed': True, 'requests_byte_identical_to_native': True,
                'native_return_values_identical': True, 'diagnostic_rows': 30,
                'prompt_or_response_content_logged': False, 'model_inference_calls': 0, 'paid_requests': 0,
                'adapter_sha256': sha(ROOT/'tools/paper_unigoal_call_diagnostics.py')}
            frozen(ROOT/'results/unigoal_call_diagnostics_gate.json', result)
            print(json.dumps(result), flush=True)
            return
        result = {'passed': True, 'requests_per_mode': 30, 'model_inference_calls': 0,
            'paid_requests': 0, 'model_and_generation_fields_byte_identical': True,
            'image_PNG_bytes_and_message_order_unchanged': True,
            'fixture_response_and_native_return_values_identical': True,
            'TCP_connections_fresh': 30, 'TCP_connections_pooled': 1,
            'fresh_transport_fixture_seconds': seconds0, 'pooled_transport_fixture_seconds': seconds1,
            'pooling': pooling, 'native_llm_source_sha256': sha(source),
            'adapter_sha256': sha(ROOT/'tools/paper_unigoal_client_pool.py'),
            'speedup_scope': 'Local CPU/HTTP overhead only; not actual model or navigation throughput',
            'navigation_end_to_end_equivalence_pending': True}
        frozen(ROOT/'results/unigoal_client_pool_gate.json', result)
        print(json.dumps(result, indent=2), flush=True)
    finally:
        server.shutdown(); server.server_close()


if __name__ == '__main__':
    main()
