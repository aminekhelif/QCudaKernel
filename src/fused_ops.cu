#include "matrix_ops.hpp"
#include <cuda_runtime.h>

#ifndef TILE_SIZE
#define TILE_SIZE 16
#endif

__global__ void fusedGemmReluKernel(const baseType* __restrict__ A,
                                    const baseType* __restrict__ B,
                                    baseType* __restrict__ C,
                                    int M, int N, int K)
{
    __shared__ baseType Asub[TILE_SIZE][TILE_SIZE];
    __shared__ baseType Bsub[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float accum = 0.0f;
    for (int t = 0; t<(K+TILE_SIZE-1)/TILE_SIZE; t++) {
        int Acol = t*TILE_SIZE + threadIdx.x;
        int Brow = t*TILE_SIZE + threadIdx.y;

        Asub[threadIdx.y][threadIdx.x] = (row<M && Acol<K)? A[row*K + Acol]:toBaseType(0.0f);
        Bsub[threadIdx.y][threadIdx.x] = (Brow<K && col<N)? B[Brow*N + col]:toBaseType(0.0f);

        __syncthreads();
        #pragma unroll
        for (int i=0; i<TILE_SIZE; i++) {
            accum += mulVal(Asub[threadIdx.y][i], Bsub[i][threadIdx.x]);
        }
        __syncthreads();
    }

    if (row<M && col<N) {
        C[row*N+col] = reluVal(toBaseType(accum));
    }
}

extern "C" void launchFusedGemmReluKernel(const baseType* d_A, const baseType* d_B, baseType* d_C,
                                          int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(TILE_SIZE,TILE_SIZE);
    dim3 gridDim((N+TILE_SIZE-1)/TILE_SIZE,(M+TILE_SIZE-1)/TILE_SIZE);
    fusedGemmReluKernel<<<gridDim, blockDim,0,stream>>>(d_A, d_B, d_C, M, N, K);
}