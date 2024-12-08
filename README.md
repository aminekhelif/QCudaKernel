# Quantum Based Cuda Kernel

This project demonstrates advanced CUDA kernels for deep learning workloads and provides:

- High-performance GEMM (classical and Tensor Core)
- Mixed precision (FP16) support
- Fused GEMM+ReLU kernel
- Quantum-inspired search kernel
- Auto-tuning logic
- CUDA Graphs for reduced launch overhead
- Comprehensive unit tests (GoogleTest)
- Performance logging and plotting (CSV and matplotlib)

## Requirements

- NVIDIA GPU with CUDA support (Compute Capability >= 7.0 for Tensor Cores)
- CUDA Toolkit installed
- CMake >= 3.10
- Python 3 and `matplotlib` (`pip install matplotlib`)

## Usage

```bash
./run_all.sh
