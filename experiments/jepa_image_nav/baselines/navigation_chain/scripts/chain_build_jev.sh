#!/usr/bin/env bash
set -euo pipefail
ROOT=/root/autodl-tmp/jepa_instance_bench_20261016
PREFIX="$ROOT/runtime/ros_workspace/.pixi/envs/default"
export CMAKE_PREFIX_PATH="$ROOT/native_deps:$PREFIX:${CMAKE_PREFIX_PATH:-}"
export LD_LIBRARY_PATH="$ROOT/native_deps/lib:$PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PKG_CONFIG_PATH="$ROOT/native_deps/lib/pkgconfig:$PREFIX/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
cmake -S "$ROOT/repos/osqp" -B "$ROOT/repos/osqp/build" \
  -DCMAKE_INSTALL_PREFIX="$ROOT/native_deps" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DBUILD_SHARED_LIBS=ON -DUNITTESTS=OFF -DOSQP_VERSION=0.6.3
cmake --build "$ROOT/repos/osqp/build" -j4
cmake --install "$ROOT/repos/osqp/build"
cmake -S "$ROOT/repos/osqp-eigen" -B "$ROOT/repos/osqp-eigen/build" \
  -DCMAKE_INSTALL_PREFIX="$ROOT/native_deps" -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
  -DCMAKE_PREFIX_PATH="$ROOT/native_deps;$PREFIX" -DBUILD_TESTING=OFF
cmake --build "$ROOT/repos/osqp-eigen/build" -j4
cmake --install "$ROOT/repos/osqp-eigen/build"
cd "$ROOT/repos/jev-objnav/jev_obj"
if [ -f build/Makefile ]; then
  cmake --build build --target plan_env_generate_messages_cpp trajectory_manager_generate_messages_cpp lkh_mtsp_solver_generate_messages_cpp -j4
fi
catkin_make -DPYTHON_EXECUTABLE="$PREFIX/bin/python" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_PREFIX_PATH="$ROOT/native_deps;$PREFIX" -j4
find devel -type f -executable -name '*node*' -o -name 'exploration_node' -o -name 'traj_server'
