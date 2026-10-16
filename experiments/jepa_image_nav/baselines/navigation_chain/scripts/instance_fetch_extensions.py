"""Sources and public cached examples for task-aligned extension benchmarks."""
import json
from instance_fetch_assets import ROOT, source, download

if __name__ == '__main__':
    for repo, name in [('openai/CLIP', 'CLIP'), ('EmbodiedBench/EmbodiedBench', 'EmbodiedBench'),
                       ('rai-opensource/vlfm', 'vlfm'), ('bagh2178/GC-VLN', 'GC-VLN')]:
        try:
            source(repo, name)
        except Exception as exc:
            print(json.dumps({'repo': repo, 'error_class': type(exc).__name__, 'http_status': getattr(exc, 'code', None)}), flush=True)
    for file in ['goal_cache/iin/train_embeddings/1S7LAXRdDqK_CLIP_goat_embedding.pkl',
                 'goal_cache/ovon/category_name_clip_embeddings.pkl']:
        try:
            download('https://huggingface.co/datasets/axel81/goat-bench/resolve/6df4daf962da14d5a57315e9c80f2969f814e6d9/' + file,
                     file.split('/')[-1])
        except Exception as exc:
            print(json.dumps({'file': file, 'error_class': type(exc).__name__}), flush=True)
