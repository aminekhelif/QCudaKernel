#include "config.hpp"
#include "utils.cuh"
#include <cuda_runtime.h>
#include <cmath>

__global__ void quantumSearchKernel(const float* data, int size, float target, int* resultIndex) {
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    float score = -1.0f;
    int idxBest = -1;
    if (tid < size) {
        float diff = fabsf(data[tid]-target);
        if (diff<0.1f) {
            float val = 1.0f/(diff+1e-4f);
            if (val>score) {
                score=val; idxBest=tid;
            }
        }
    }

    float bestVal=score;
    int bestIdx=idxBest;

    for(int offset=16; offset>0; offset>>=1) {
        float otherVal = __shfl_down_sync(0xffffffff, bestVal, offset);
        int otherIdx = __shfl_down_sync(0xffffffff, bestIdx, offset);
        if (otherVal > bestVal) {
            bestVal=otherVal; bestIdx=otherIdx;
        }
    }

    __shared__ float blockVals[32];
    __shared__ int blockIdxs[32];
    int lane = threadIdx.x%32;
    int warpId = threadIdx.x/32;

    if(lane==0) {
        blockVals[warpId]=bestVal;
        blockIdxs[warpId]=bestIdx;
    }
    __syncthreads();

    if(threadIdx.x<32) {
        float bVal=blockVals[threadIdx.x];
        int bIdx=blockIdxs[threadIdx.x];
        for (int offset=16; offset>0; offset>>=1) {
            float oVal = __shfl_down_sync(0xffffffff,bVal,offset);
            int oIdx  = __shfl_down_sync(0xffffffff,bIdx,offset);
            if (oVal>bVal){bVal=oVal;bIdx=oIdx;}
        }
        if (threadIdx.x==0 && bIdx>=0) atomicCAS(resultIndex,-1,bIdx);
    }
}

extern "C" void launchQuantumSearch(const float* d_data, int size, float target, int* d_resultIndex, cudaStream_t stream) {
    int threads=256;
    int blocks=(size+threads-1)/threads;
    quantumSearchKernel<<<blocks,threads,0,stream>>>(d_data,size,target,d_resultIndex);
}