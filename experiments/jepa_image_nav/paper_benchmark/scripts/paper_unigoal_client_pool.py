"""Reuse only the native local HTTP client; no response cache or request changes."""
import contextlib
import time
from urllib.parse import urlparse


@contextlib.contextmanager
def pooled_clients(module):
    original = module.OpenAI
    clients = {}
    report = {'factory_calls': 0, 'clients_constructed': 0, 'factory_seconds': 0.,
        'response_cache': False, 'request_parameters_changed': False,
        'scope': 'Per-episode local-loopback HTTP connection reuse; no cross-episode navigation memory'}
    def factory(*args, **kwargs):
        started = time.monotonic()
        report['factory_calls'] += 1
        try:
            # Native LLM/VLM use precisely these2 constructor options. Anything
            # else retains its original constructor semantics rather than guessing.
            if args or set(kwargs) != {'api_key', 'base_url'}:
                return original(*args, **kwargs)
            if urlparse(str(kwargs['base_url'])).hostname not in {'localhost', '127.0.0.1', '::1'}:
                return original(**kwargs)
            key = (str(kwargs['base_url']), kwargs['api_key'])
            if key not in clients:
                clients[key] = original(**kwargs)
                report['clients_constructed'] += 1
            return clients[key]
        finally:
            report['factory_seconds'] += time.monotonic()-started
    module.OpenAI = factory
    try:
        yield report
    finally:
        module.OpenAI = original
        for client in clients.values():
            client.close()

