"""Preserve appended reviewed findings when refreshing the live generated snapshot."""
from pathlib import Path
import json,hashlib,time
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
marker='<!-- PRESERVED_STAGE_HISTORY_START -->'
files=[R/'reports/frontier_transfer_review.md',R.parent/'reports/navigation_capability_merged.md']
audit=[]
for path in files:
 old=path.read_text()
 assert marker not in old
 # Manual chronological entries follow the generated current-execution snapshot.
 pos=old.index('### 当前执行快照')
 split=old.index('\n## ',pos)
 backup=path.with_name(path.name+'.pre_history_preserve_20260911')
 assert not backup.exists();backup.write_text(old)
 new=old[:split]+'\n'+marker+'\n'+old[split:]
 path.write_text(new)
 assert old==new.replace('\n'+marker+'\n','',1)
 audit.append({'path':str(path),'original_sha256':hashlib.sha256(old.encode()).hexdigest(),'preserved_history_chars':len(old[split:]),'backup':str(backup)})
p=R/'summarize_navigation.py';old=p.read_text();backup=p.with_name('summarize_navigation_pre_history_preserve.py')
assert not backup.exists();backup.write_text(old)
before=" body=path.read_text().split('### 当前执行快照')[0].split('## 阶段机制审计记录')[0].rstrip()"
after=""" original=path.read_text()
 history_marker='<!-- PRESERVED_STAGE_HISTORY_START -->'
 generated,sep,history=original.partition(history_marker)
 assert sep, 'Missing preserved stage history marker; refusing to rewrite '+str(path)
 body=generated.split('### 当前执行快照')[0].split('## 阶段机制审计记录')[0].rstrip()"""
assert old.count(before)==1
new=old.replace(before,after)
assert new.count(' path.write_text(body)')==1
new=new.replace(' path.write_text(body)'," path.write_text(body+'\\n\\n'+history_marker+history)")
new=new.replace('尚需阶段审阅、补充训练种子、关键迁移基线、多布局训练和新拓扑验证；部分观测是否加入由证据决定。',
'剩余工作以报告末尾带日期的阶段审阅和队列为准，勿将历史计划当作当前未完成项；部分观测是否加入由证据决定。')
compile(new,str(p),'exec');p.write_text(new)
(R/'reports/report_history_preservation_audit.json').write_text(json.dumps({'updated_unix':time.time(),'passed':True,'files':audit,'summary_script_backup':str(backup)},indent=2))
print('HISTORY_PRESERVATION_PATCHED',flush=True)
