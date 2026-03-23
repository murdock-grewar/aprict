#include <cuda/atomic>

#include "test_subdir/test_config.h" // Provides Test_Config_Sum

template<typename T>
__global__ void sumsqs(Test_Config_Sum config, T* vec, int len, T* output_num) {
    /* ASSUMPTION: this kernel is called with grid/block dims of
                   y and z lengths equal to 1. */
    __shared__ cuda::atomic<T, cuda::thread_scope_block> sum;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) { return; }

    bool block_leader = (tid == 0);
    if (block_leader) {
        sum = 0;
    }

    T myterm = vec[tid];
    myterm *= myterm;
    if (config.alternate_sign) {
        if (tid%2 == 1) {
            myterm *= -1;
        }
    }
    if (tid == 0) {
        myterm *= (T)config.multiply_first_term;
    }
    sum += myterm;

    if (block_leader) {
        atomicAdd(output_num, sum);
    }
}

extern "C"
__global__ void sumsqs_float(Test_Config_Sum config, float* vec, int len, float* output_num) {
    /* ASSUMPION: this kernel is called with grid/block dims of
                   y and z lengths equal to 1. */
    __shared__ cuda::atomic<float, cuda::thread_scope_block> sum;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= len) { return; }

    float myterm = vec[tid];
    myterm *= myterm;
    bool block_leader = (tid == 0);
    if (config.alternate_sign) {
        if (tid%2 == 1) {
            myterm *= -1;
        }
    }
    if (tid == 0) {
        myterm *= (float)config.multiply_first_term;
    }
    sum += myterm;

    if (block_leader) {
        atomicAdd(output_num, sum);
    }
}
