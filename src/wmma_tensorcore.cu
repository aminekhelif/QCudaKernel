#include "wmma_gemm.hpp"
#include "config.hpp"
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;

__global__ void wmmaGemmKernel(const baseType* __restrict__ A,
                               const baseType* __restrict__ B,
                               baseType* __restrict__ C,
                               int M, int N, int K)
{
    // Each block computes one 16x16 tile of C
    // M, N, K should be multiples of 16 for simplicity.
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y);
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x);

    wmma::fragment<wmma::matrix_a, 16,16,16, baseType, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, 16,16,16, baseType, wmma::col_major> bFrag;
    wmma::fragment<wmma::accumulator, 16,16,16, float> cFrag;

    wmma::fill_fragment(cFrag, 0.0f);

    int globalRow = warpM * 16;
    int globalCol = warpN * 16;

    for (int k=0; k<K; k+=16) {
        const baseType* A_tile = &A[globalRow*K + k];
        const baseType* B_tile = &B[k*N + globalCol];

        wmma::load_matrix_sync(aFrag, A_tile, K);
        wmma::load_matrix_sync(bFrag, B_tile, N);
        wmma::mma_sync(cFrag, aFrag, bFrag, cFrag);
    }

    baseType *C_tile = &C[globalRow*N + globalCol];
    wmma::store_matrix_sync(C_tile, cFrag, N, wmma::mem_row_major);
}

extern "C" void launchWmmaGemmKernel(const baseType* d_A, const baseType* d_B, baseType* d_C,
                                     int M, int N, int K, cudaStream_t stream) {
    dim3 blockDim(1,1);
    dim3 gridDim(N/16,M/16);
    wmmaGemmKernel<<<gridDim, blockDim, 0, stream>>>(d_A, d_B, d_C, M, N, K);
}