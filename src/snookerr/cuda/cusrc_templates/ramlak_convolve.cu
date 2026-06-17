#include <cuda_fp16.h>

/* Note on the 'half' type, used for 16-bit floating point precision:
 * The texture writing function, 'surf3Dwrite' or 'surf2Dwrite', doesn't support
 * input values of type 'half'. So, I wrote the function below which has a template
 * specialisation to the 'half' type. It allows us to write halfs to the texture
 * by temporarily converting them to unsigned shorts, using CUDA's fp16 API included
 * with #include <cuda_fp16.h>.
 */

#define PI 3.1415926535898f
#define PISQ 9.8696044011f
template<typename T>
__device__ T fcast(const half f)
{
    return (T)__half2float(f);
}
template<typename T>
__device__ T fcast(const float f)
{
    return (T)f;
}
template<typename T>
__device__ T fcast(const double f)
{
    return (T)f;
}
template<>
__device__ half fcast<half>(const half f)
{
    return f;
}
template<>
__device__ half fcast<half>(const float f)
{
    return __float2half(f);
}
template<>
__device__ half fcast<half>(const double f)
{
    return __double2half(f);
}

template <typename Tp, unsigned int N>
__global__ void convolve_with_ramlak_x(
    Tp*             data,
    const uint3     shape,
    const ulong3    strides_bytes,  // byte strides between elements. This method will ignore the x stride, since that is assumed to be equal to sizeof(Tp).
    const float     physpixwid
)
{
    // Assumes a block size of (BX, 1, 1).
    // Assumes a grid size of (GX, GY, GZ). Preferably, GX = 1 and BX = N, but BX can only be up to 1024 on current architectures. We require BX*GX >= N.
    // Assumes that the smallest stride in memory is along the convolution axis

    const unsigned long data_offset_bytes = blockIdx.y * strides_bytes.y + blockIdx.z * strides_bytes.z;

    // The whole thing will be dealt with in blocks of up to 1024 elements long, since is the maximum number of threads in a CUDA block.
    unsigned int num_groups =  (N + (blockDim.x-1))/blockDim.x;  // integer division is intentional
    // Flag whether to skip this thread. Note: this thread will still assist in loading data, so it can't return just yet.
    const unsigned int xpos = threadIdx.x + blockIdx.x * blockDim.x;
    const bool skip_this_thread = (xpos >= shape.x);

    // Within each block, the threads will load their piece of the data into shared memory
    Tp val = (Tp)0;
    Tp subval;
    unsigned int ramlak_index;
    float c, sincpiterm, sincpihalfterm;
    __shared__ Tp row[1024];  // Only need up to blockDim.x, which is no more than 1024
    unsigned int data_address_group = (data_offset_bytes/sizeof(Tp) + threadIdx.x);
    const unsigned int data_address = data_address_group + blockIdx.x * blockDim.x;
    for (unsigned short groupid = 0; groupid < num_groups; groupid++) {
        // Load the data from the group
        row[threadIdx.x] = data[data_address_group];
        __syncthreads();
        // Add the contribution of the ramp filter applied to the group
        unsigned int imax = min(((groupid+1)*blockDim.x), shape.x);
        unsigned int j = 0;
        subval = (Tp)0;
        for (unsigned int i = groupid*blockDim.x; i < imax; i++) {
            // Compute the kernel coefficient: c
            ramlak_index = abs((int)i - (int)xpos);
            // sincpi = (ramlak_index == 0) ? 1 : sin(PI*ramlak_index)/(PI*ramlak_index);
            sincpiterm = (ramlak_index == 0) ? PISQ : 0;  // This would be multiplied by 2, except that part is subtracted by the sincpihalfterm at ramlak_index == 0. I've absorbed that into this term instead of into the second term.
            sincpihalfterm = (ramlak_index%2 == 0) ? 0 : (
                -4.f/(ramlak_index*ramlak_index)
            );
            c = sincpiterm + sincpihalfterm;
            // Add the coefficient multiplied by the component
            subval += (row[j] * fcast<Tp>(c));
            j++;
        }
        val += subval;
        // Prepare for the next group.
        data_address_group += blockDim.x;
        __syncthreads();
    }
    if (!skip_this_thread) {
        data[data_address] = val/physpixwid;
    }
}

