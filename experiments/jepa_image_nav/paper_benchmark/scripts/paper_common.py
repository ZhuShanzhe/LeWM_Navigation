"""Versioned, task-local comparison contract; official tracks keep their own rules."""
from paper_freeze import ROOT, frozen

PROFILE = {'resolution': [480, 640], 'hfov': 79, 'camera_height': .88,
           'radius': .18, 'height': 1.4, 'allow_sliding': False}
CONTRACT = {
    'version': 1, 'track': 'HSSD official-val43 derived controlled comparison',
    'profile': PROFILE, 'depth_clip_m': [0., 5.], 'forward_m': .25,
    'turn_degrees': 30, 'look_degrees': 30, 'max_actions': 500,
    'goal_photographs': 'Original frozen photographs; identical per episode across methods',
    'place_primary': 'Voluntary STOP and geodesic distance to goal position <1m',
    'instance_primary': 'Voluntary STOP, geodesic to frozen approach set <0.5m, same-instance current pixels >=0.025',
    'instance_SPL': 'Primary success weighted by shortest path to the same frozen approach set; not official object-region SPL',
    'forced_stop': 'No synthetic STOP for common image-goal tracks; budget exhaustion is unsuccessful termination',
    'available_actions': ['STOP', 'forward', 'left', 'right', 'look_up', 'look_down'],
    'native_action_interfaces': {'FGPrompt': 'STOP/forward/left/right',
        'DEBiT': 'STOP/forward/left/right', 'UniGoal': 'STOP/forward/left/right',
        'GOAT': 'Native subtask STOP6; global abort0 is terminal failure; forward/turn/look preserved'},
    'policy_inputs': {'FGPrompt': ['RGB', 'goal_RGB', 'previous_action'],
        'DEBiT': ['RGB', 'goal_RGB', 'previous_action'],
        'GOAT': ['RGB', 'goal_RGB_or_category', 'ideal_relative_GPS', 'compass', 'previous_action'],
        'UniGoal': ['RGB', 'depth', 'goal_RGB', 'ideal_odometry']},
    'comparison_limits': [
        'Equal physical conditions do not imply identical used modalities or native action heads',
        'GOAT versus UniGoal is a whole-system comparison, not an isolated equal-input encoder comparison',
        'FGPrompt and DEBiT are matched RGB-only place baselines, not native instance-navigation evidence',
        'All43 official validation scenes are evaluation-only; no result-driven resampling or tuning',
        'Pretraining exposure must be reported per checkpoint; scene split alone does not certify unseen pretraining'],
    'statistics': {'primary': 'Equal scene-weighted SR and task-defined SPL',
        'secondary': ['episode-weighted SR', 'distance/detour/instance-asset slices', 'collision rate',
                      'latency', 'peak VRAM', 'search/stop/instance-confusion diagnostics'],
        'uncertainty': 'Paired scene-cluster bootstrap10000, seed20261017; no independent-episode CI assumption',
        'replications': 'Report actual policy decoding and seeds; do not label evaluation repeats as training seeds'},
    'official_tracks': {
        'gibson_imagegoal': 'All original14-scene validation episodes, native author task/sensor/goal-camera/metric configuration',
        'hssd_objectnav': 'Original HSSD ObjectNav validation release; native Habitat measures and source goal schema'},
    'release_gate': ['all selected cases accounted', 'independent trace/score audit',
        'same realized environment hashes for matched methods', 'official-track interface regression',
        'preserved legacy result hashes', 'secret and restricted asset scan', 'verified remote Git commit']}

if __name__ == '__main__':
    frozen(ROOT/'protocol/common_v1.json', CONTRACT)
    print('COMMON_CONTRACT_FROZEN')
