import json,gdown
from pathlib import Path
items=gdown.download_folder('https://drive.google.com/drive/folders/1r31os0d4-rR0mdHc7OlY_e5nh3XT4r4e',skip_download=True,quiet=True)
rows=[dict(id=i.id,path=i.path) for i in items]
Path('/root/autodl-tmp/lewm_research/round12h_20260909/baseline_drive_index.json').write_text(json.dumps(rows,indent=2))
print(rows)
