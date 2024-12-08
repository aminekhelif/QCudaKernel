#include <cuda_runtime.h>

extern "C" void runWithCudaGraph(void(*kernelFunc)(cudaStream_t), cudaStream_t stream=0) {
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    cudaStream_t capStream;
    cudaStreamCreate(&capStream);

    cudaStreamBeginCapture(capStream, cudaStreamCaptureModeGlobal);
    kernelFunc(capStream);
    cudaStreamEndCapture(capStream, &graph);
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graphExec);
    cudaStreamDestroy(capStream);
}