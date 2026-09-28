"""Build-only adaptation: exclude unused ROS branch from the official ZMQ simulator path.

This does not provide a replacement sensor algorithm. Generated source retains
its upstream GPLv3 header. Diff and unchanged-function hashes are audit artifacts.
"""
import argparse, difflib, hashlib, json
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('source',type=Path)
p.add_argument('output',type=Path)
a=p.parse_args()
original=a.source.read_text()
modified=original
for inc in ['#include <livox_ros_driver2/msg/custom_msg.hpp>', '#include <rclcpp/rclcpp.hpp>']:
    assert modified.count(inc)==1
    modified=modified.replace(inc, '// ROS-only include omitted in simulation-only build')
start=modified.index('static void\nprojectToSphericalGrid(')
end=modified.index('// Normalize grid:',start)
modified=modified[:start]+modified[end:]
start=modified.index('class LidarDepthPub : public rclcpp::Node {')
end=modified.index('// ── ZMQ source mode',start)
modified=modified[:start]+modified[end:]
start=modified.index('    // ── Default: ROS2 Livox source')
modified=modified[:start]+'    std::cerr << "Simulation-only build requires --sim" << std::endl;\n    return 2;\n}\n'
assert 'rclcpp::' not in modified and 'livox_ros_driver2::' not in modified
assert original[original.index('static void projectRawPointsToGrid('):original.index('// ── Main')] == modified[modified.index('static void projectRawPointsToGrid('):modified.index('// ── Main')]
a.output.parent.mkdir(parents=True,exist_ok=True)
a.output.write_text(modified)
a.output.with_suffix('.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile=str(a.source),tofile=str(a.output))))
protected=original[original.index('static void projectRawPointsToGrid('):original.index('// ── Main')]
a.output.with_suffix('.audit.json').write_text(json.dumps({'scope':'ROS-only branch excluded; entire ZMQ projection/receive/publish block unchanged byte-for-byte','source_sha256':hashlib.sha256(original.encode()).hexdigest(),'adapted_sha256':hashlib.sha256(modified.encode()).hexdigest(),'protected_zmq_block_sha256':hashlib.sha256(protected.encode()).hexdigest(),'equivalence_scope':'source-block identity; not yet a closed-loop timing validation'},indent=2))
