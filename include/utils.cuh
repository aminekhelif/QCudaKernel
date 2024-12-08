#ifndef UTILS_CUH
#define UTILS_CUH

__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

#endif // UTILS_CUH