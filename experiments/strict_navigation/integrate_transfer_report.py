from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'summarize_navigation.py';s=p.read_text();needle=" runpy.run_path(str(R/'geometry_result_report.py'))"
if "runpy.run_path(str(R/'transfer_result_report.py'))" not in s:s=s.replace(needle,needle+"\n runpy.run_path(str(R/'transfer_result_report.py'))")
needle="('geometry_results.md','几何布局能力结果')"
if "('transfer_mechanism_results.md'" not in s:s=s.replace(needle,needle+",('transfer_mechanism_results.md','迁移机制与研究方向判断')")
p.write_text(s)
p=R/'geometry_result_report.py';s=p.read_text().replace('其4个验证几何布局比较已排队。','其4个验证几何布局已完成，具体差值与解释见transfer_mechanism_results.md。');p.write_text(s)
p=R/'reports/retrieval_adaptation.md';s=p.read_text().replace('正式GPU冒烟与验证已加入唯一串行队列；尚无正式检索性能结论。','正式GPU冒烟与100例短长目标验证均已完成；结果与附加价值判断见transfer_mechanism_results.md。');p.write_text(s)
