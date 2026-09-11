"""Same frozen oracle evaluator; only waypoint switching tolerance changes8->16px."""
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
source=(R/'eval_oracle_waypoints.py').read_text()
before="mode=os.environ.get('LEWM_ROUTE_MODE','waypoint'))"
after="mode=os.environ.get('LEWM_ROUTE_MODE','waypoint'),threshold=16.0)"
assert source.count(before)==1
exec(compile(source.replace(before,after),str(R/'eval_oracle_waypoints_t16.py'),'exec'),globals())
