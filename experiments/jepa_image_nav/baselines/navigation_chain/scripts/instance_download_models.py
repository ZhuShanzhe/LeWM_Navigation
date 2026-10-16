"""Publisher model assets for full native navigation chains."""
import concurrent.futures as cf
import json
from instance_fetch_assets import ROOT, get, download

def guarded(url, filename):
    try:
        download(url, filename)
    except Exception as exc:
        print(json.dumps({'filename': filename, 'error_class': type(exc).__name__,
                          'http_status': getattr(exc, 'code', None)}), flush=True)

if __name__ == '__main__':
    repo = 'timm/resnet50_clip.openai'
    revision = json.loads(get('https://huggingface.co/api/models/' + repo))['sha']
    (ROOT / 'results/clip_revision.json').write_text(json.dumps({'repository': repo, 'revision': revision}))
    jobs = [
        ('https://huggingface.co/' + repo + '/resolve/' + revision + '/open_clip_model.safetensors', 'clip_RN50_openai.safetensors'),
        ('https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth', 'sam_vit_h_4b8939.pth'),
        ('https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth', 'groundingdino_swint_ogc.pth'),
        ('https://dl.fbaipublicfiles.com/detectron2/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl', 'mask_rcnn_R_50_FPN_3x.pkl'),
    ]
    with cf.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda x: guarded(*x), jobs))
