from pathlib import Path
p=Path('/root/autodl-tmp/lewm_research/round12h_20260909/download_all.py')
s=p.read_text().replace("import json,time,urllib.request","import os,json,time,urllib.request")
s=s.replace("('.tar.zst','.h5','.tar.gz')","('.tar.zst','.h5','.h5.zst','.tar.gz')")
s=s.replace("    elif name.endswith('.h5'):", """    elif name.endswith('.h5.zst'):
     outpath=dest/path.name[:-4]
     with path.open('rb') as inp,outpath.open('wb') as out:
      zstd.ZstdDecompressor().copy_stream(inp,out)
    elif name.endswith('.h5'):""")
s=s.replace("['pusht','cube','reacher']","os.environ.get('LEWM_DOWNLOAD_ENVS','pusht,cube,reacher').split(',')")
p.write_text(s)
