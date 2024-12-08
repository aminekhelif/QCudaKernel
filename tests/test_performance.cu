#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <vector>
#include <cmath>
#include <fstream>
#include <chrono>
#include "../include/config.hpp"
#include "../include/matrix_ops.hpp"

extern "C" void launchMatMulKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);
extern "C" void launchWmmaGemmKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);
extern "C" void launchFusedGemmReluKernel(const baseType*, const baseType*, baseType*,int,int,int,cudaStream_t);

static const int M=512;
static const int N=512;
static const int Kdim=512;

class PerformanceTest : public ::testing::Test {
protected:
    std::vector<float> h_A, h_B, h_Cref;
    baseType *d_A,*d_B,*d_C;
    int sizeA, sizeB, sizeC;
    static std::ofstream resultsFile;
    static bool opened;

    virtual void SetUp() {
        sizeA = M*Kdim;
        sizeB = Kdim*N;
        sizeC = M*N;
        h_A.resize(sizeA);
        h_B.resize(sizeB);
        h_Cref.resize(sizeC);

        srand(1234);
        for(int i=0;i<sizeA;i++) h_A[i]=(float)rand()/RAND_MAX;
        for(int j=0;j<sizeB;j++) h_B[j]=(float)rand()/RAND_MAX;

        CUDA_CHECK(cudaMalloc(&d_A,sizeA*sizeof(baseType)));
        CUDA_CHECK(cudaMalloc(&d_B,sizeB*sizeof(baseType)));
        CUDA_CHECK(cudaMalloc(&d_C,sizeC*sizeof(baseType)));

        std::vector<baseType> h_A_fp(sizeA), h_B_fp(sizeB);
        for(int i=0;i<sizeA;i++) h_A_fp[i]=toBaseType(h_A[i]);
        for(int j=0;j<sizeB;j++) h_B_fp[j]=toBaseType(h_B[j]);

        CUDA_CHECK(cudaMemcpy(d_A,h_A_fp.data(),sizeA*sizeof(baseType),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_B,h_B_fp.data(),sizeB*sizeof(baseType),cudaMemcpyHostToDevice));

        // CPU Reference
        for(int r=0; r<M; r++){
            for(int c=0;c<N;c++){
                double sum=0.0;
                for(int k=0;k<Kdim;k++){
                    sum+= (double)h_A[r*Kdim+k]* (double)h_B[k*N+c];
                }
                h_Cref[r*N+c]=(float)sum;
            }
        }

        if(!opened) {
            resultsFile.open("results.csv",std::ios::out);
            resultsFile << "Kernel,TimeMs,MaxError\n";
            opened=true;
        }
    }

    virtual void TearDown() {
        CUDA_CHECK(cudaFree(d_A));
        CUDA_CHECK(cudaFree(d_B));
        CUDA_CHECK(cudaFree(d_C));
    }

    void checkAndRecord(const std::string &kernelName, double timeMs) {
        std::vector<baseType> h_Cfp(sizeC);
        CUDA_CHECK(cudaMemcpy(h_Cfp.data(),d_C,sizeC*sizeof(baseType),cudaMemcpyDeviceToHost));
        double maxErr=0.0;
        for(int i=0;i<100;i++) {
            int r=rand()%M;int c=rand()%N;
            float ctest;
#if defined(USE_HALF_PRECISION)
            ctest=__half2float(h_Cfp[r*N+c]);
#else
            ctest=h_Cfp[r*N+c];
#endif
            double diff = fabs(ctest-h_Cref[r*N+c]);
            if(diff>maxErr) maxErr=diff;
        }
        resultsFile << kernelName << "," << timeMs << "," << maxErr << "\n";
    }

    void runKernel(void(*kernelFunc)(const baseType*,const baseType*,baseType*,int,int,int,cudaStream_t),
                   const std::string &name) {
        CUDA_CHECK(cudaMemset(d_C,0,sizeC*sizeof(baseType)));
        auto start=std::chrono::high_resolution_clock::now();
        kernelFunc(d_A,d_B,d_C,M,N,Kdim,0);
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(end-start).count();
        checkAndRecord(name,ms);
    }
};

std::ofstream PerformanceTest::resultsFile;
bool PerformanceTest::opened=false;

TEST_F(PerformanceTest, ClassicGEMM) {
    runKernel(launchMatMulKernel,"ClassicGEMM");
}

TEST_F(PerformanceTest, TensorCoreGEMM) {
#if ENABLE_TENSOR_CORES
    runKernel(launchWmmaGemmKernel,"TensorCoreGEMM");
#else
    GTEST_SKIP() << "Tensor Cores not enabled.";
#endif
}

TEST_F(PerformanceTest, FusedGEMMReLU) {
    runKernel(launchFusedGemmReluKernel,"FusedGEMMReLU");
}