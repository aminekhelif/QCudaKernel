#include <gtest/gtest.h>
extern "C" void launchQuantumSearch(const float*, int, float, int*, cudaStream_t);
extern "C" void autoTuneKernelParams();
extern "C" int getSelectedTileSize();

TEST(UtilsTest, AutoTuner) {
    autoTuneKernelParams();
    int tile = getSelectedTileSize();
    ASSERT_GT(tile,0);
}

TEST(UtilsTest, QuantumSearch) {
    int size=1024;
    std::vector<float> data(size);
    for(int i=0;i<size;i++) data[i]=(float)rand()/RAND_MAX;
    int specialIdx = 512;
    data[specialIdx]=0.50001f;

    float *d_data;
    int *d_result;
    cudaMalloc(&d_data,size*sizeof(float));
    cudaMalloc(&d_result,sizeof(int));

    cudaMemcpy(d_data,data.data(),size*sizeof(float),cudaMemcpyHostToDevice);
    int initVal=-1;
    cudaMemcpy(d_result,&initVal,sizeof(int),cudaMemcpyHostToDevice);

    launchQuantumSearch(d_data,size,0.5f,d_result,0);
    cudaDeviceSynchronize();

    int found;
    cudaMemcpy(&found,d_result,sizeof(int),cudaMemcpyDeviceToHost);
    cudaFree(d_data);
    cudaFree(d_result);

    ASSERT_EQ(found,specialIdx);
}