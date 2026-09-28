#!/usr/bin/env bash
# Standalone sensor acceptance only; never starts a robot controller.
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "Usage: $0 /absolute/path/to/JEPLO /absolute/path/to/new-output" >&2; exit 2; fi
REPO="$(realpath "$1")"
OUT="$(realpath -m "$2")"
SRC="$(cd "$(dirname "$0")" && pwd)"
if [ -e "$OUT" ]; then echo "Use a new output directory to preserve previous evidence." >&2; exit 2; fi
mkdir -p "$OUT"
python "$SRC/audit_official_profile.py" --repo "$REPO" --output "$OUT/source_model_gate.json"
python "$SRC/official_profile_negative_tests.py" "$REPO" "$SRC/../configs/official_readme_profile.json" "$OUT/profile_negative_tests.json"
python "$SRC/prepare_official_sensor.py" "$REPO/deployment/lidar_depth_pub/lidar_depth_pub.cpp" "$OUT/lidar_depth_pub_sim.cpp"
g++ -std=c++17 -O2 -I "$OUT" "$SRC/official_sensor_unit.cpp" -lzmq -lpthread -o "$OUT/sensor_unit"
"$OUT/sensor_unit" | tee "$OUT/sensor_unit.txt"
g++ -std=c++17 -O2 "$OUT/lidar_depth_pub_sim.cpp" -lzmq -lpthread -o "$OUT/lidar_depth_pub_sim"
python "$SRC/official_sensor_live_test.py" --binary "$OUT/lidar_depth_pub_sim" --output "$OUT/sensor_live_test.json"
