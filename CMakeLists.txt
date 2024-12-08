cmake_minimum_required(VERSION 3.10)
project(CUDA_Advanced LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_HALF_PRECISION "Use half precision" ON)
option(USE_TENSOR_CORES "Use Tensor Cores" ON)

find_package(CUDA REQUIRED)
set(CUDA_ARCH "70")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_${CUDA_ARCH} -O3")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SRC_FILES
    src/main.cu
    src/matrix_mul.cu
    src/wmma_tensorcore.cu
    src/fused_ops.cu
    src/quantum_search.cu
    src/auto_tuner.cu
    src/cuda_graphs.cu
    src/utils.cu
)

add_executable(cuda_advanced ${SRC_FILES})
target_link_libraries(cuda_advanced cuda)

enable_testing()
add_subdirectory(tests)