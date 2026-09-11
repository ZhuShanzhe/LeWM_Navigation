from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'summarize_navigation.py';s=p.read_text()
needle="   boots=delta[rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)*100"
replacement="   pair_seed=int.from_bytes(hashlib.sha256((key+'|'+alt+'|'+stratum).encode()).digest()[:8],'little')\n   pair_rng=np.random.default_rng(pair_seed)\n   boots=delta[pair_rng.integers(0,len(delta),size=(10000,len(delta)))].mean(1)*100"
if needle in s:s=s.replace(needle,replacement)
needle="save(R/'reports/navigation_results_audit.json',payload)"
if "runpy.run_path(str(R/'head_result_report.py'))" not in s:
 s=s.replace(needle,needle+"\nif (R/'heads/temporal3072/summary.json').exists():\n import runpy\n runpy.run_path(str(R/'head_result_report.py'))")
p.write_text(s)
p=R/'reports/temporal_head_adaptation.md';s=p.read_text()
start=s.index('## 已完成与待完成');end=s.index('## 来源',start)
s=s[:start]+"""## 已完成与待完成

小规模CPU训练与控制接入检查已通过。首个模型的正式编码、时间头和打乱标签头各20轮训练、GPU接口检查已完成，导航验证陆续完成；具体完整组及快照见“时间评价头阶段结果”，不从冒烟推断性能。另两个独立模型的重复与4个验证几何布局已安排，尚不构成完成结果。最终预留病例仍未使用。

"""+s[end:];p.write_text(s)
