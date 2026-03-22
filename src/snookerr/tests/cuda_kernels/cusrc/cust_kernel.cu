#include <cuda/atomic>

#include "Config_Sum.h" // The contents of this file will be injected before
                        // passing this code to CuPy for compilation.
                        // Note that CuPy is planned to begin caching the
                        // RawModuesls generated with 'name_expressions=[...]'
                        // but will still not hash included dependencies.

/* In order for cupy to load the kernels from a module, it needs to know the
 * names. These names are mangled when a .cu file is compiled into a .cubin
 * file, unless the option 'extern "C"' is used before the function, in which
 * case cupy somehow knows how to extract the function with
 * RawModule.get_function().
 * However, template functions are incompatible with 'extern "C"', and so it is
 * not possible to get cupy to load a precompiled cubin with a template
 * function, and then actually use that template function, in an automatic
 * fashion. (In theory it is still possible, by manually inspecting the mangled
 * names and passing them to RawModule.get_function()).
//extern "C" __global__
template<typename T>
__global__
void sumsqs(Config_Sum config, T* vec, int len, T* output_num) {
*/
template<typename T>
__global__ void sumsqs(Config_Sum config, T* vec, int len, T* output_num) {
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
__global__ void sumsqs_float(Config_Sum config, float* vec, int len, float* output_num) {
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
