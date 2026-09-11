from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
p=R/'summarize_navigation.py';s=p.read_text()
needle=" runpy.run_path(str(R/'head_result_report.py'))"
if "runpy.run_path(str(R/'geometry_result_report.py'))" not in s:s=s.replace(needle,needle+"\n runpy.run_path(str(R/'geometry_result_report.py'))")
needle="('retrieval_adaptation.md','训练轨迹检索对照')"
s=s.replace(needle,needle+",('geometry_results.md','几何布局能力结果'),('geometry_collection_protocol.md','匹配采集与多布局训练准备')" if "('geometry_results.md'" not in s else needle)
p.write_text(s)
