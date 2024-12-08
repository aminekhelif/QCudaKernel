#ifndef CONFIG_HPP
#define CONFIG_HPP

#if defined(USE_HALF_PRECISION)
#include <cuda_fp16.h>
typedef __half baseType;
#else
typedef float baseType;
#endif

#if defined(USE_TENSOR_CORES)
#define ENABLE_TENSOR_CORES 1
#else
#define ENABLE_TENSOR_CORES 0
#endif

#define CUDA_CHECK(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#endif // CONFIG_HPP