// Test harness only. Includes adapted upstream GPLv3 source, obtained separately.
#define main upstream_sensor_main
#include "lidar_depth_pub_sim.cpp"
#undef main
#include <cassert>
int main() {
    assert(configureFov("25x60"));
    float grid[kMaxGridSize];
    // A retained central point at 1 m; a closer point at index 1 is skipped by stride 4.
    float pts[15]={1,0,0, .5f,0,0, 0,0,0, 0,0,0, 1.5f,0,0};
    projectRawPointsToGrid(pts,5,grid,4);
    float minimum=*std::min_element(grid,grid+gGridSize);
    assert(std::abs(minimum-1.f)<1e-6);
    normalizeGrid(grid);
    assert(std::abs(*std::min_element(grid,grid+gGridSize)-.5f)<1e-6);
    projectRawPointsToGrid(pts,5,grid,1);
    assert(std::abs(*std::min_element(grid,grid+gGridSize)-.5f)<1e-6);
    std::vector<float> history(5*gGridSize,1.f),out(gGridSize);
    history[gGridSize+37]=.25f;
    accumulateMin(history.data(),5,out.data());
    assert(out[37]==.25f && out[38]==1.f);
    bool mask[kMaxGridSize]={false}; mask[37]=true;
    applyOcclusionMask(out.data(),mask);
    assert(out[37]==1.f);
    std::cout << "PASS stride4, stride1 contrast, normalization, five-scan min, missing-value mask\n";
}
