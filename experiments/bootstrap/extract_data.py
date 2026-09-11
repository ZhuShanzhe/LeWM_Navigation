import hashlib,json,tarfile,time
from pathlib import Path
import zstandard
root=Path('/root/autodl-tmp/lewm_research/data')
archive=root/'tworoom.tar.zst'
with archive.open('rb') as raw,zstandard.ZstdDecompressor().stream_reader(raw) as decompressed,tarfile.open(fileobj=decompressed,mode='r|') as tar:
 for member in tar:
  resolved=(root/member.name).resolve()
  if not resolved.is_relative_to(root.resolve()) or member.issym() or member.islnk(): raise ValueError('Unsafe archive member '+member.name)
  print(member.name,member.size,flush=True)
  tar.extract(member,path=root,filter='data')
print('EXTRACT_COMPLETE',flush=True)
