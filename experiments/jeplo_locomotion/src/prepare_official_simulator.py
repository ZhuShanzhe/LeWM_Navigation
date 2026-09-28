"""Exclude unavailable optional RGB-depth render callback for LiDAR-only builds.
Original JEPLO source is kept intact; emit an auditable patch for the isolated copy.
"""
import difflib,hashlib,json,sys
from pathlib import Path
path=Path(sys.argv[1]);original=path.read_text()
modified=original.replace('#include "depth_camera.h"','// Optional RGB-depth camera excluded; LiDAR implementation unchanged.')
start=modified.index('    // Depth camera: initialize and set up render callback')
end=modified.index('    // start physics thread',start)
modified=modified[:start]+'    if (param::config.enable_depth) {\n        std::cerr << "This build supports LiDAR only; --depth is unavailable." << std::endl;\n        std::exit(2);\n    }\n\n'+modified[end:]
assert original[:original.index('// run event loop')].replace('#include "depth_camera.h"','') == modified[:modified.index('// run event loop')].replace('// Optional RGB-depth camera excluded; LiDAR implementation unchanged.','')
path.with_suffix('.original.cc').write_text(original)
path.with_suffix('.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),modified.splitlines(True),fromfile='upstream/main.cc',tofile='lidar-only/main.cc')))
path.write_text(modified)
path.with_suffix('.audit.json').write_text(json.dumps({'source_sha256':hashlib.sha256(original.encode()).hexdigest(),'modified_sha256':hashlib.sha256(modified.encode()).hexdigest(),'scope':'only optional RGB-depth callback/include excluded; physics thread, terrain and LiDAR source unchanged','not_confirmed_paper_mujoco_version':True},indent=2))
