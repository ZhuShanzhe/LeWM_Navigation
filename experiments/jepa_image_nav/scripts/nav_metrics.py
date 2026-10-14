"""Metric functions; callers supply observed STOP and actual executed path length."""
def compute_metrics(stopped, final_distance, initial_distance, executed_length, collisions, steps, instance_visible=None, wrong_instance=False):
    # Distance is place geodesic distance or distance to nearest validated approach view.
    threshold=1.0 if instance_visible is None else .5
    success=bool(stopped and final_distance<threshold and (instance_visible is None or instance_visible))
    efficiency=initial_distance/max(initial_distance,executed_length,1e-8)
    progress=max(0.,1.-final_distance/max(initial_distance,1e-8))
    return {'success':int(success),'spl':int(success)*efficiency,'softspl':progress*efficiency,
            'collision_rate':collisions/max(steps,1),'wrong_instance_stop':int(stopped and wrong_instance),
            'final_distance_m':final_distance,'executed_length_m':executed_length,'steps':steps}
if __name__=='__main__':
    assert compute_metrics(True,.3,5,10,1,20)['spl']==.5
    assert compute_metrics(True,.3,5,10,0,20,instance_visible=False)['success']==0
    assert compute_metrics(False,.1,5,5,0,20)['success']==0
    print('METRIC_UNIT_CHECKS_PASSED')
