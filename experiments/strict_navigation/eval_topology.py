"""Read-only adaptation of strict evaluator to the separately audited topology simulator."""
from pathlib import Path
import json,os,gymnasium as gym
from topology_env import TopologyEnv
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
catalog=json.loads((R/'topology_dev_v2/catalog.json').read_text())
assert catalog['passed']
gym.register('swm/TopologyDiagnostic-v0',entry_point=TopologyEnv)
src=(R/'eval_strict.py').read_text()
replacements={
 "if 'init_value' in case_data:":"if 'topology_spec' in case_data:",
 "self.env=gym.make('swm/TwoRoom-v1',init_value=case_data['init_value']).unwrapped":"self.env=gym.make('swm/TopologyDiagnostic-v0',spec=case_data['topology_spec']).unwrapped",
 "kw['init_value']=case_data['init_value']":"kw['env_name']='swm/TopologyDiagnostic-v0'\n  kw['spec']=case_data['topology_spec']",
 "'across_wall'":"'different_room'",
}
for before,after in replacements.items():
 assert before in src,before
 src=src.replace(before,after)
exec(compile(src,str(R/'eval_topology.py'),'exec'),globals())
