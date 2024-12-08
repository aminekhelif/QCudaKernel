#ifndef WMMA_GEMM_HPP
#define WMMA_GEMM_HPP

#include "config.hpp"
#include <mma.h>

using namespace nvcuda;

struct WmmaGemmConfig {
    static const int M = 16;
    static const int N = 16;
    static const int K = 16;
};

#endif // WMMA_GEMM_HPP