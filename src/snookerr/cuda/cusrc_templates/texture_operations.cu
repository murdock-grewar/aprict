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


// Multi-surface operations
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
