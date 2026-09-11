"""Inject privileged goal routing without altering environment goals or flushing action buffers."""
from pathlib import Path
R=Path('/root/autodl-tmp/lewm_research/strict_nav_20260910')
injection="""from oracle_waypoint_router import OracleWaypointRouter
_route_instances=[]
_route_base_action=swm.policy.WorldModelPolicy.get_action
def routed_action(self,info,*a,**kw):
 if not hasattr(self,'_oracle_router'):
  saved_torch=torch.get_rng_state();saved_np=np.random.get_state();saved_python=random.getstate()
  saved_cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
  try:
   self._oracle_router=OracleWaypointRouter(case_data['topology_spec'],cases,mode=os.environ.get('LEWM_ROUTE_MODE','waypoint'))
  finally:
   torch.set_rng_state(saved_torch);np.random.set_state(saved_np);random.setstate(saved_python)
   if saved_cuda is not None:torch.cuda.set_rng_state_all(saved_cuda)
  _route_instances.append(self._oracle_router)
 return _route_base_action(self,self._oracle_router.transform(info),*a,**kw)
swm.policy.WorldModelPolicy.get_action=routed_action
"""
driver=(R/'eval_topology.py').read_text()
needle="exec(compile(src,str(R/'eval_topology.py'),'exec'),globals())"
assert driver.count(needle)==1
replacement="assert src.count('orig_action=swm.policy.WorldModelPolicy.get_action')==1\nsrc=src.replace('orig_action=swm.policy.WorldModelPolicy.get_action',injection+'orig_action=swm.policy.WorldModelPolicy.get_action')\n"+needle+"\nassert len(_route_instances)==1\n(out/'oracle_waypoint_metadata.json').write_text(json.dumps(_route_instances[0].report(),indent=2))\n"
driver=driver.replace(needle,replacement)
exec(compile(driver,str(R/'eval_oracle_waypoints.py'),'exec'),globals())
