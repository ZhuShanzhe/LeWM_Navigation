"""Read-only native SDK call timing/finish telemetry; no request kwargs modifications."""
import contextlib
import hashlib
import json
import sys
import time


@contextlib.contextmanager
def call_diagnostics(folder):
    from openai.resources.chat.completions import Completions
    original = Completions.create
    def measured(client, *args, **kwargs):
        started = time.monotonic()
        row = {'started_unix_s': time.time(), 'model': kwargs.get('model'), 'status': 'interrupted',
            'request_sha256': hashlib.sha256(json.dumps(kwargs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
        frame = sys._getframe(1); callers = []
        while frame is not None and len(callers) < 6:
            if '/src/' in frame.f_code.co_filename:
                callers.append({'function': frame.f_code.co_name, 'line': frame.f_lineno})
            frame = frame.f_back
        row['native_callers'] = callers
        try:
            response = original(client, *args, **kwargs)
            usage = getattr(response, 'usage', None)
            row.update(status='complete', prompt_tokens=getattr(usage, 'prompt_tokens', None),
                completion_tokens=getattr(usage, 'completion_tokens', None),
                finish_reasons=[getattr(c, 'finish_reason', None) for c in response.choices])
            extra = getattr(response, 'model_extra', None) or {}
            row['server_timing_if_available'] = {k: extra[k] for k in ['total_duration', 'load_duration',
                'prompt_eval_duration', 'eval_duration'] if k in extra and isinstance(extra[k], (float, int))}
            return response
        except Exception as exc:
            row.update(status='runtime_failure', error=type(exc).__name__)
            raise
        finally:
            row['seconds'] = time.monotonic()-started; row['finished_unix_s'] = time.time()
            with (folder/'model_call_diagnostics.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
    Completions.create = measured
    try:
        yield
    finally:
        Completions.create = original
