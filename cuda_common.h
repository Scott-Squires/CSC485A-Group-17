/**
 * Standard macros that can be useful for error checking.
 * https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__ERROR.html
 */
#pragma once

#include <cuda.h>

#define CUDA_CALL(exp)                                       \
    do {                                                     \
        cudaError res = (exp);                               \
        if(res != cudaSuccess) {                             \
            printf("Error at %s:%d\n %s\n",                  \
                __FILE__,__LINE__, cudaGetErrorString(res)); \
           exit(EXIT_FAILURE);                               \
        }                                                    \
    } while(0)

#define CHECK_ERROR(msg)                                             \
    do {                                                             \
        cudaError_t err = cudaGetLastError();                        \
        if(cudaSuccess != err) {                                     \
            printf("Error (%s) at %s:%d\n %s\n",                     \
                (msg), __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                      \
        }                                                            \
    } while (0)