#include <cstdio>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include "config.hpp"

extern "C" void launchMatMulKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);
extern "C" void launchWmmaGemmKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);
extern "C" void launchFusedGemmReluKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);
extern "C" void launchQuantumSearch(const float*, int, float,int*, cudaStream_t);
extern "C" void autoTuneKernelParams();
extern "C" int getSelectedTileSize();
extern "C" void runWithCudaGraph(void(*kernelFunc)(cudaStream_t), cudaStream_t);

static const int M=1024;
static const int N=1024;
static const int Kdim=1024;

static __global__ void dummyKernel(){}

void testKernel(cudaStream_t stream) {
    dummyKernel<<<1,1,0,stream>>>();
}

int main() {
    autoTuneKernelParams();
    int tileSize=getSelectedTileSize();
    printf("Selected tile size: %d\n", tileSize);

    std::vector<float> h_A(M*Kdim),h_B(Kdim*N), h_C(M*N);
    for (int i=0;i<M*Kdim;i++) h_A[i]= (float)rand()/RAND_MAX;
    for (int i=0;i<Kdim*N;i++) h_B[i]=(float)rand()/RAND_MAX;

    baseType *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A,M*Kdim*sizeof(baseType)));
    CUDA_CHECK(cudaMalloc(&d_B,Kdim*N*sizeof(baseType)));
    CUDA_CHECK(cudaMalloc(&d_C,M*N*sizeof(baseType)));

    std::vector<baseType> h_A_fp(M*Kdim), h_B_fp(Kdim*N);
    for (int i=0;i<M*Kdim;i++) h_A_fp[i]=toBaseType(h_A[i]);
    for (int i=0;i<Kdim*N;i++) h_B_fp[i]=toBaseType(h_B[i]);

    CUDA_CHECK(cudaMemcpy(d_A,h_A_fp.data(),M*Kdim*sizeof(baseType),cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B,h_B_fp.data(),Kdim*N*sizeof(baseType),cudaMemcpyHostToDevice));

#if ENABLE_TENSOR_CORES
    printf("Using Tensor Cores WMMA GEMM...\n");
    auto start=std::chrono::high_resolution_clock::now();
    launchWmmaGemmKernel(d_A,d_B,d_C,M,N,Kdim);
#else
    printf("Using Classic GEMM...\n");
    auto start=std::chrono::high_resolution_clock::now();
    launchMatMulKernel(d_A,d_B,d_C,M,N,Kdim);
#endif
    CUDA_CHECK(cudaDeviceSynchronize());
    auto end=std::chrono::high_resolution_clock::now();
    double ms=std::chrono::duration<double,std::milli>(end-start).count();
    printf("GEMM completed in %.3f ms.\n",ms);

    CUDA_CHECK(cudaMemcpy(h_C.data(),d_C,M*N*sizeof(baseType),cudaMemcpyDeviceToHost));

#if defined(USE_HALF_PRECISION)
    std::vector<float> h_Cf(M*N);
    for (int i=0;i<M*N;i++) {
        h_Cf[i]=__half2float(h_C[i]);
    }
    double maxErr=0.0;
    for (int i=0;i<100;i++){
        int r=rand()%M;int c=rand()%N;
        double ref=0.0;
        for (int k=0;k<Kdim;k++) ref+= (double)h_A[r*Kdim+k]* (double)h_B[k*N+c];
        maxErr = fmax(maxErr,fabs(h_Cf[r*N+c]-ref));
    }
#else
    double maxErr=0.0;
    for (int i=0;i<100;i++){
        int r=rand()%M;int c=rand()%N;
        double ref=0.0;
        for (int k=0;k<Kdim;k++) ref+= (double)h_A[r*Kdim+k]* (double)h_B[k*N+c];
        maxErr = fmax(maxErr,fabs((double)h_C[r*N+c]-ref));
    }
#endif
    printf("Max verification error: %e\n",maxErr);

    // Quantum-inspired search
    std::vector<float> h_dataSearch(M*N);
    for (int i=0;i<M*N;i++) h_dataSearch[i]=(float)rand()/RAND_MAX;
    float *d_dataSearch; int *d_resultIndex;
    CUDA_CHECK(cudaMalloc(&d_dataSearch,M*N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_resultIndex,sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_dataSearch,h_dataSearch.data(),M*N*sizeof(float),cudaMemcpyHostToDevice));
    int initVal=-1;
    CUDA_CHECK(cudaMemcpy(d_resultIndex,&initVal,sizeof(int),cudaMemcpyHostToDevice));
    launchQuantumSearch(d_dataSearch,M*N,0.5f,d_resultIndex);
    CUDA_CHECK(cudaDeviceSynchronize());

    int foundIndex;
    CUDA_CHECK(cudaMemcpy(&foundIndex,d_resultIndex,sizeof(int),cudaMemcpyDeviceToHost));
    printf("Quantum-inspired search found index: %d\n",foundIndex);

    // Fused GEMM+ReLU
    CUDA_CHECK(cudaMemset(d_C,0,M*N*sizeof(baseType)));
    launchFusedGemmReluKernel(d_A,d_B,d_C,M,N,Kdim);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Fused GEMM+ReLU done.\n");

    // CUDA Graphs demonstration
    runWithCudaGraph(testKernel);
    printf("CUDA Graph run completed.\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_dataSearch));
    CUDA_CHECK(cudaFree(d_resultIndex));

    return 0;
}