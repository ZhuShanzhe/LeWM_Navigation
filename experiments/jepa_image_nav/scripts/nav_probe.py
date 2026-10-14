import habitat_sim, json, numpy as np, magnum as mn
from pathlib import Path
from PIL import Image
root=Path('/root/autodl-tmp/jepa_vln_data_20261014')
c=habitat_sim.SimulatorConfiguration(); c.scene_dataset_config_file=str(root/'raw/hssd-pilot/hssd-hab.scene_dataset_config.json');c.scene_id='102344115'
c.enable_physics=True
a=habitat_sim.agent.AgentConfiguration();a.height=1.4;a.radius=.18
sensors=[]
for name,t in [('rgb',habitat_sim.SensorType.COLOR),('depth',habitat_sim.SensorType.DEPTH),('semantic',habitat_sim.SensorType.SEMANTIC)]:
    s=habitat_sim.CameraSensorSpec();s.uuid=name;s.sensor_type=t;s.resolution=[256,256];s.position=mn.Vector3(0,1.25,0);s.hfov=mn.Deg(90);sensors.append(s)
a.sensor_specifications=sensors
sim=habitat_sim.Simulator(habitat_sim.Configuration(c,[a]))
settings=habitat_sim.NavMeshSettings();settings.set_defaults();settings.agent_radius=.18;settings.agent_height=1.4;settings.include_static_objects=True
ok=sim.recompute_navmesh(sim.pathfinder,settings)
mgr=sim.get_rigid_object_manager();objs=mgr.get_objects_by_handle_substring('')
print('NAVMESH',ok,'AREA',sim.pathfinder.navigable_area,'OBJECTS',len(objs),flush=True)
for i,obj in enumerate(objs.values(),1): obj.semantic_id=i
agent=sim.get_agent(0);st=agent.get_state();st.position=sim.pathfinder.get_random_navigable_point();agent.set_state(st)
o=sim.get_sensor_observations();Image.fromarray(o['rgb'][:,:,:3]).save(root/'reports/hssd_pilot_rgb.png')
print('RGB',o['rgb'].shape,'SEMANTIC_IDS',np.unique(o['semantic']).tolist(),flush=True)
print('OBJECT_SAMPLE',[(o.handle,list(o.translation),o.semantic_id) for o in list(objs.values())[:3]],flush=True)
sim.close()
