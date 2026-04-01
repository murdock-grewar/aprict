#include <cuda_fp16.h>

/* Note on the 'half' type, used for 16-bit floating point precision:
 * The texture writing function, 'surf3Dwrite' or 'surf2Dwrite', doesn't support
 * input values of type 'half'. So, I wrote the function below which has a template
 * specialisation to the 'half' type. It allows us to write halfs to the texture
 * by temporarily converting them to unsigned shorts, using CUDA's fp16 API included
 * with #include <cuda_fp16.h>.
 */
template<typename T>
__device__ void write_to_surf3D(
    T                       value,
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ surf3Dwrite(value, surface,  idx.z * sizeof(T), idx.y, idx.x); }

template<>
__device__ void write_to_surf3D<half>(
    half                    value,
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ write_to_surf3D<unsigned short>(__half_as_ushort(value), surface, idx); }


// Utility functions used repeatedly
__device__ uint3 get_vol_idx()
{
    return uint3({
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    });
}
__device__ bool out_of_max_bounds(uint3 idx, uint3 stop_idx)
{
    if (idx.x >= stop_idx.x) { return true; }
    if (idx.y >= stop_idx.y) { return true; }
    if (idx.z >= stop_idx.z) { return true; }
    return false;
}
__device__ bool out_of_minmax_bounds(uint3 idx, uint3 start_idx, uint3 stop_idx)
{
    if (out_of_max_bounds(idx, stop_idx)) { return true; }
    if (idx.x < start_idx.x) { return true; }
    if (idx.y < start_idx.y) { return true; }
    if (idx.z < start_idx.z) { return true; }
    return false;
}

//// SINGLE-TEXTURE/SURFACE ARITHMETIC ////
template <typename T>
__global__ void fill_surface(
    const T                     value,
    cudaSurfaceObject_t         surface,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    write_to_surf3D(value, surface, idx);
}

