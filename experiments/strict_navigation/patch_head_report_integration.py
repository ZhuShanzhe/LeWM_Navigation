from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'summarize_navigation.py';s=p.read_text()
needle=" body+='\\n\\n### 当前执行快照\\n\\n'"
insert=""" for filename,title in [('head_mechanism_results.md','时间评价头：阶段验证结果'),('retrieval_adaptation.md','训练轨迹检索对照')]:
  extra_note=R/'reports'/filename
  if extra_note.exists():body+='\\n\\n### '+title+'\\n\\n'+extra_note.read_text().split('\\n',1)[1].strip()
"""
assert needle in s
if "('head_mechanism_results.md'" not in s:s=s.replace(needle,insert+needle)
s=s.replace('最佳当前验证值66%只高4个百分点','上述规划参数对照的最佳验证值66%只高4个百分点')
p.write_text(s)
