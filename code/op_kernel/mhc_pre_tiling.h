// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcPreTilingData {
    uint64_t positionCount;
    uint32_t n;
    uint32_t d;
    uint32_t width;
    uint32_t rowCount;
    uint32_t tileLength;
    uint32_t hasGamma;
    float invWidth;
    float normEps;
    float hcEps;
};
