#ifndef MATRIX_OPS_HPP
#define MATRIX_OPS_HPP

#include "config.hpp"

inline __host__ __device__ baseType toBaseType(float f) {
#if defined(USE_HALF_PRECISION)
    return __float2half(f);
#else
    return f;
#endif
}

inline __device__ float mulVal(baseType a, baseType b) {
#if defined(USE_HALF_PRECISION)
    return __half2float(a)*__half2float(b);
#else
    return a*b;
#endif
}

inline __device__ baseType addVal(baseType a, baseType b) {
#if defined(USE_HALF_PRECISION)
    return __float2half(__half2float(a)+__half2float(b));
#else
    return a+b;
#endif
}

inline __device__ baseType reluVal(baseType x) {
#if defined(USE_HALF_PRECISION)
    return (__half2float(x) > 0.0f) ? x : __float2half(0.0f);
#else
    return (x > 0.0f) ? x : 0.0f;
#endif
}

#endif // MATRIX_OPS_HPP