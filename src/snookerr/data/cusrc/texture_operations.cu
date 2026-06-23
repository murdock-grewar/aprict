#include <cuda/atomic>
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

template<typename T>
__device__ T read_from_surf3D(
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ return surf3Dread<T>(surface, idx.z * sizeof(T), idx.y, idx.x); }

template<>
__device__ half read_from_surf3D<half>(
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ return __ushort_as_half(read_from_surf3D<unsigned short>(surface, idx)); }

template<typename T>
__device__ T read_from_surf2DLayered(
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ return surf2DLayeredread<T>(surface, idx.z * sizeof(T), idx.y, idx.x); }

template<>
__device__ half read_from_surf2DLayered<half>(
    cudaSurfaceObject_t     surface,
    uint3                   idx
)
{ return __ushort_as_half(read_from_surf2DLayered<unsigned short>(surface, idx)); }


// General math functions
#define PI 3.1415926535898f

template <typename T>
__device__ T powint(T base, short exponent) {
    // Protect against negative values
    T ans = 1.;
    if (exponent < 0) {
        for (unsigned short e = 0; e < -exponent; e++) {
            ans /= base;
        }
    } else {
        for (unsigned short e = 0; e < exponent; e++) {
            ans *= base;
        }
    }
    return ans;
}

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

//// SINGLE-TEXTURE/SURFACE OPERATIONS ////
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

template <typename T> // type of the volumed data. E.g. half or float.
__global__ void softmask_surface_to_cylinder_xy(
    const float                 relrad_start,
    cudaSurfaceObject_t         surface,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    // Compute radius of cylinder, relative to 1 being the maximum
    float rrel = sqrt(
            (powint((float)idx.x - (surfaceDimensions.x-1)/2.f, 2) +
             powint((float)idx.y - (surfaceDimensions.y-1)/2.f, 2)))
        / (float)(min(surfaceDimensions.x, surfaceDimensions.y)) * 2.f;
    // Return early if inside the unsoftened region of the cylinder
    if (rrel < relrad_start) { return; }
    // Scale rrel: [relrad_start, 1] --> [0, 1]
    rrel = min((rrel-relrad_start)/(1.f-relrad_start), 1.f);
    // Get the multiplier
    T multiplier = min(1.f, max(0.f, (1+cos(rrel * PI))/2.f));
    // Read the existing value, apply the multiplier and write it back
    T value = read_from_surf3D<T>(surface, idx);
    value *= multiplier;
    write_to_surf3D(value, surface, idx);
}

template <typename T> // type of the volumed data. E.g. half or float.
__global__ void multiplicatively_invert_surface(
    cudaSurfaceObject_t         surface,
    uint3                       surfaceDimensions,
    const bool                  ignore_zeros,
    const float                 eps  // used to replace zeros with small positive values.
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    // Read the existing value
    T value = read_from_surf3D<T>(surface, idx);
    if (ignore_zeros && (value == (T)0)) { return; }
    if (eps > 0.f) {
        value = value + (T)(eps * exp(-(float)value/eps));
    }
    value = powint(value, -1);
    write_to_surf3D(value, surface, idx);
}

template <typename T>
__global__ void multiply_surface(
    const T                     value,
    cudaSurfaceObject_t         surface,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    T existing_value = read_from_surf3D<T>(surface, idx);
    write_to_surf3D(value * existing_value, surface, idx);
}

template <typename T>  // type of both the surface and the output array
__global__ void slice_nth_axis_to_3d_array(
        cudaSurfaceObject_t surf,
        const unsigned int arr_shape_x,
        const unsigned int arr_shape_y,
        const unsigned int arr_shape_z,
        const unsigned int sl_start,
        const unsigned int sl_stop,
        const unsigned char axis_index_to_slice,
        const unsigned long output_array_stride_x,  // measured in bytes
        const unsigned long output_array_stride_y,
        const unsigned long output_array_stride_z,
        const bool surf_is_layered,
        T* output_array
)
{
    uint3 idx = uint3({
        threadIdx.x + blockDim.x * blockIdx.x,
        threadIdx.y + blockDim.y * blockIdx.y,
        threadIdx.z + blockDim.z * blockIdx.z
    });
    if (idx.x >= arr_shape_x) { return; }
    if (idx.y >= arr_shape_y) { return; }
    if (idx.z >= arr_shape_z) { return; }
    int i = idx.x;
    int j = idx.y;
    int k = idx.z;

    // Select which index to slice
    int* ijk;
    if (axis_index_to_slice == 0)      { ijk = &i; idx.x += sl_start; }
    else if (axis_index_to_slice == 1) { ijk = &j; idx.y += sl_start; }
    else if (axis_index_to_slice == 2) { ijk = &k; idx.z += sl_start; }
    if (*ijk >= sl_stop - sl_start) { return; }

    T val;
    if (surf_is_layered) {
        val = read_from_surf2DLayered<T>(surf, idx);
    } else {
        val = read_from_surf3D<T>(surf, idx);
    }
    output_array[
            (i*output_array_stride_x + j*output_array_stride_y + k*output_array_stride_z)/sizeof(T)
        ] = val;
}

template <typename T>  // type of both the surface and the output array
__global__ void sum_along_nth_axis_to_2d_array(
        cudaSurfaceObject_t surf,
        const unsigned int surf_shape_x,
        const unsigned int surf_shape_y,
        const unsigned int surf_shape_z,
        const unsigned char axis_index_to_sum,
        const unsigned long output_array_stride_x,  // measured in bytes
        const unsigned long output_array_stride_y,
        const bool surf_is_layered,
        T* output_array
)
{
    uint3 idx = uint3({
        threadIdx.x + blockDim.x * blockIdx.x,
        threadIdx.y + blockDim.y * blockIdx.y,
        threadIdx.z + blockDim.z * blockIdx.z
    });
    if (idx.x >= surf_shape_x) { return; }
    if (idx.y >= surf_shape_y) { return; }
    if (idx.z >= surf_shape_z) { return; }
    int i = idx.x;
    int j = idx.y;
    int k = idx.z;

    // Get the index of this thread in the 2D output into the array
    unsigned long iarr, jarr;
    if (axis_index_to_sum == 0) { iarr = idx.y; jarr = idx.z; }
    if (axis_index_to_sum == 1) { iarr = idx.x; jarr = idx.z; }
    if (axis_index_to_sum == 2) { iarr = idx.x; jarr = idx.y; }

    T val;
    if (surf_is_layered) {
        val = read_from_surf2DLayered<T>(surf, idx);
    } else {
        val = read_from_surf3D<T>(surf, idx);
    }
    atomicAdd(&output_array[
            (iarr*output_array_stride_x + jarr*output_array_stride_y)/sizeof(T)
        ], val);
}

// Multi-surface operations
template <typename T1, typename T2>
__global__ void inner_product_3D_surfaces(
    double*                     output,
    cudaSurfaceObject_t         surface1,
    cudaSurfaceObject_t         surface2,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    const bool is_block_leader = (threadIdx.x == 0) && (threadIdx.y == 0) && (threadIdx.z == 0);
    cuda::atomic<double, cuda::thread_scope_block> dotprod;
    __syncthreads();
    if (!(out_of_max_bounds(idx, surfaceDimensions))) {
        dotprod += (double)read_from_surf3D<T1>(surface1, idx) * (double)read_from_surf3D<T2>(surface2, idx);
    }
    if (is_block_leader) {
        atomicAdd(output, dotprod);
    }
}

template <typename T1, typename T2, typename TO>
__global__ void multiply_surfaces(
    cudaSurfaceObject_t         surface1,
    cudaSurfaceObject_t         surface2,
    cudaSurfaceObject_t         surfaceOut,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    TO val = read_from_surf3D<T1>(surface1, idx);
    val *= (TO)read_from_surf3D<T2>(surface2, idx);
    write_to_surf3D(val, surfaceOut, idx);
}

template <typename T1, typename T2, typename TO>
__global__ void add_surfaces(
    cudaSurfaceObject_t         surface1,
    cudaSurfaceObject_t         surface2,
    cudaSurfaceObject_t         surfaceOut,
    uint3                       surfaceDimensions
)
{
    const uint3 idx = get_vol_idx();
    if (out_of_max_bounds(idx, surfaceDimensions)) { return; }
    TO val = read_from_surf3D<T1>(surface1, idx);
    val += (TO)read_from_surf3D<T2>(surface2, idx);
    write_to_surf3D(val, surfaceOut, idx);
}
