#include <cuda/atomic>
#include <cuda_fp16.h>

#include "base.h"
#include "geometry.h"
#include "FORBILD_volumes.h"


#define MAX_NUM_SHAPES_FOR_SHARED_MEMORY 300  // = 300, based on 48KB shared memory size, and about 150 bytes per Shape object at the time of writing
template<
    typename Tv,  // type of volume data. e.g. half or float or double
    typename Ts   // Can be FORBILD_Shape, which is inefficient but allows passing a bunch of different shapes at once. Or can be a specific shape type, like FORBILD_Sphere, et cetera
>
__global__ void add_FORBILD_sampling(
        const uint3            vol_shape,
        const float*            x_arr,
        const float*            y_arr,
        const float*            z_arr,
        Ts*                     shapes_glob,     // shapes array
        const unsigned int      num_shapes,      // number of shapes
        const ulong3            output_vol_strides,
        Tv*                     output_vol,
        float                   supersamp_cubic_vox_wid,
        unsigned char           supersamp_factor
)
{
    uivec3 vol_idx = uivec3(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    );
    const bool in_range = (
        (vol_idx.x < vol_shape.x) &&
        (vol_idx.y < vol_shape.y) &&
        (vol_idx.z < vol_shape.z)
    );
    const unsigned long num_threads = (
        min(vol_shape.x, blockDim.x) *
        min(vol_shape.y, blockDim.y) *
        min(vol_shape.z, blockDim.z)
    );

    // Load the first N shapes into shared memory. Process them, then load the next. Etc.
    const unsigned long tid = (
        threadIdx.x +
        threadIdx.y*blockDim.x +
        threadIdx.z*blockDim.x*blockDim.y
    );
    // Load shapes into shared memory for the thread block
    __shared__ Ts shapes_shared[MAX_NUM_SHAPES_FOR_SHARED_MEMORY];
    Tv totval = 0;
    fvec3 pos_c = fvec3(x_arr[vol_idx.x], y_arr[vol_idx.y], z_arr[vol_idx.z]);

    // In the case that all the shapes fit in the shared memory,
    // then we don't need to reload them to do the supersampling. This makes
    // for a significant optimisation for phantoms with relatively small numbers
    // of shapes, such as the FORBILD head phantom.
    const bool single_shape_batch = (num_shapes <= MAX_NUM_SHAPES_FOR_SHARED_MEMORY);
    bool not_first_batch = false;

    for (unsigned char offx = 0; offx < supersamp_factor; offx++) {
    for (unsigned char offy = 0; offy < supersamp_factor; offy++) {
    for (unsigned char offz = 0; offz < supersamp_factor; offz++) {
        unsigned int num_shapes_remaining = num_shapes;
        unsigned int num_shapes_loaded = 0;
        fvec3 pos = pos_c + fvec3(
            (offx - (supersamp_factor - 1.f)/2.f)/(float)supersamp_factor,
            (offz - (supersamp_factor - 1.f)/2.f)/(float)supersamp_factor,
            (offy - (supersamp_factor - 1.f)/2.f)/(float)supersamp_factor
        ) * supersamp_cubic_vox_wid;
        Tv val = 0;  // Default to 0 if no shapes overwrite it
        while (num_shapes_remaining > 0) {
            unsigned int num_shapes_to_load = min(
                num_shapes_remaining,
                MAX_NUM_SHAPES_FOR_SHARED_MEMORY
            );
            unsigned int num_shapes_to_load_per_thread = (
                (unsigned int)ceil((float)num_shapes_to_load / num_threads)
            );
            __syncthreads();  // Wait for all threads to finish dealing with the last round of data
            for (
                unsigned long i = tid*num_shapes_to_load_per_thread;
                i < (1+tid)*num_shapes_to_load_per_thread;
                i++
            ) {
                if (i >= num_shapes_to_load) { break; }
                if (single_shape_batch && not_first_batch) { break; }
                shapes_shared[i] = shapes_glob[i + num_shapes_loaded];
            }
            not_first_batch = true;
            __syncthreads();

            // Data has been loaded into shared memory. Now do something with it.
            // Loop through shapes IN ORDER. The last found shape determines the value.
            if (in_range) {
                for (unsigned int i = 0; i < num_shapes_to_load; i++) {
                    Ts shape = shapes_shared[i];
                    if (_in_region(pos, shape)) {
                        val = shape.rho;
                    }
                }
            }

            // Move on to the next round of shapes
            num_shapes_loaded    += num_shapes_to_load;
            num_shapes_remaining -= num_shapes_to_load;
            __syncthreads();  // Wait for all threads to finish
        }
        totval += val;
    }}}
    if (totval == (Tv)0) { return; }
    totval /= supersamp_factor*supersamp_factor*supersamp_factor;

    // Add the result atomically.
    // (using atomicAdd allows us to launch other kernels concurrently, if desired)
    // Get the index 'out_idx' of the output array that this thread should write to
    unsigned long out_idx = (
        output_vol_strides.x * vol_idx.x +
        output_vol_strides.y * vol_idx.y +
        output_vol_strides.z * vol_idx.z
    )/sizeof(Tv);
    atomicAdd(&output_vol[out_idx], totval);
}
