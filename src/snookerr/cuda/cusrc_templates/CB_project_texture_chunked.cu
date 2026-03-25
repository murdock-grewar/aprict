#include <cuda/atomic>

// Provides config_CB_project, volume_chunk_spec, det_spec, fvec3
#include "config_cb_project.h"

// Define a structure and method to compute line integrals through voxelised volumes
typedef struct {
    float x;
    float y;
    float z;
    // d{x,y,z} defines a vector (of any length) which points in the direction of the line
    float dx;
    float dy;
    float dz;
} LineIntegState;


template<
    typename Tv,
    typename Tp>  // type of projection data. e.g. float or double
__device__ Tp exact_line_integral(
        const fvec3                 v1,
        const fvec3                 v2,
        const vol_tex_chunk_spec    texchunkspec,
        cudaTextureObject_t         volume_texture,
        // A debug option. Instead of returning the line intgeral, return the number of steps taken which were through a region not contained in the chunk. This indicates the inefficient.
        const bool                  return_false_steps = false)
{
    // MODE OF OPERATION
    // This function steps between boundaries of voxels, and adds the exact line integral components.

    const fvec3 minxyz = texchunkspec.offsets;
    const fvec3 maxxyz = minxyz + texchunkspec.shape;

    // Measure distances relative to the bottom-left corner of the 000 voxel of the overall texture.
    // This way, integer values for x,y,z correspond to voxel boundaries.
    float xstart = v1.x - texchunkspec.texpos.x;
    float xend   = v2.x - texchunkspec.texpos.x;
    if ((xstart < minxyz.x) && (xend < minxyz.x)) { return 0.; }
    if ((xstart > maxxyz.x) && (xend > maxxyz.x)) { return 0.; }
    float ystart = v1.y - texchunkspec.texpos.y;
    float yend   = v2.y - texchunkspec.texpos.y;
    if ((ystart < minxyz.y) && (yend < minxyz.y)) { return 0.; }
    if ((ystart > maxxyz.y) && (yend > maxxyz.y)) { return 0.; }
    float zstart = v1.z - texchunkspec.texpos.z;
    float zend   = v2.z - texchunkspec.texpos.z;
    if ((zstart < minxyz.z) && (zend < minxyz.z)) { return 0.; }
    if ((zstart > maxxyz.z) && (zend > maxxyz.z)) { return 0.; }

    Tp accum_val = 0;

    // Initialise the struct
    LineIntegState lis;
    lis.x = xstart;
    lis.y = ystart;
    lis.z = zstart;
    lis.dx = xend - xstart;
    lis.dy = yend - ystart;
    lis.dz = zend - zstart;

    //
    int xdir, ydir, zdir;
    if (lis.dx < 0) { xdir = -1; } else if (lis.dx == 0) { xdir = 0; } else { xdir = 1; }
    if (lis.dy < 0) { ydir = -1; } else if (lis.dy == 0) { ydir = 0; } else { ydir = 1; }
    if (lis.dz < 0) { zdir = -1; } else if (lis.dz == 0) { zdir = 0; } else { zdir = 1; }

    // Initalise nextxbound, nextybound, nextzbound
    // These have to be floats because the end of the integral is at xend, yend, zend.
    float nextxbound, nextybound, nextzbound; // each records the next x/y/z value which would trigger a crossing into a new voxel

    float length_per_step = norm3d(lis.dx, lis.dy, lis.dz);
    float steps_to_end = 1.f; // By definition of lis.dx,dy,dz, a single step goes from start to end.
    float steps_to_take;
    float steps_to_take_x, steps_to_take_y, steps_to_take_z;
    int idx_x, idx_y, idx_z;
    //// Skip to the first intersection with the volume (plus an extra voxel for safety)
    float advance_steps = 0;
    if (xdir < 0) { advance_steps = max(advance_steps, (maxxyz.x + 1./2. - lis.x)/lis.dx); }
    if (xdir > 0) { advance_steps = max(advance_steps, (minxyz.x - 1./2. - lis.x)/lis.dx); }
    if (ydir < 0) { advance_steps = max(advance_steps, (maxxyz.y + 1./2. - lis.y)/lis.dy); }
    if (ydir > 0) { advance_steps = max(advance_steps, (minxyz.y - 1./2. - lis.y)/lis.dy); }
    if (zdir < 0) { advance_steps = max(advance_steps, (maxxyz.z + 1./2. - lis.z)/lis.dz); }
    if (zdir > 0) { advance_steps = max(advance_steps, (minxyz.z - 1./2. - lis.z)/lis.dz); }
    if (advance_steps > 0) {
        lis.x += advance_steps * lis.dx;
        lis.y += advance_steps * lis.dy;
        lis.z += advance_steps * lis.dz;
        steps_to_end -= advance_steps;
    }
    ///////
    while (steps_to_end > 0) { // Loop until 'steps_to_end' becomes nonpositive, which would indicate that we have reached the end of the line integral
        // Find the critical values of x, y, z
        // Initially, I had planned to calculate this once at the beginning, and only update the necessary x/y/z values as necessary.
        // However, that approach would miss corner cases where the ray hits two boundaries at once. It's safer to totally recalculate these each loop.
        if (xdir <= 0) {
            nextxbound = ceil(lis.x) - 1;
        } else {
            nextxbound = floor(lis.x) + 1;
        }
        if (ydir <= 0) {
            nextybound = ceil(lis.y) - 1;
        } else {
            nextybound = floor(lis.y) + 1;
        }
        if (zdir <= 0) {
            nextzbound = ceil(lis.z) - 1;
        } else {
            nextzbound = floor(lis.z) + 1;
        }
        if ((xdir <= 0) && (nextxbound < minxyz.x)) { break; }
        if ((xdir >  0) && (nextxbound > maxxyz.x)) { break; }
        if ((ydir <= 0) && (nextybound < minxyz.y)) { break; }
        if ((ydir >  0) && (nextybound > maxxyz.y)) { break; }
        if ((zdir <= 0) && (nextzbound < minxyz.z)) { break; }
        if ((zdir >  0) && (nextzbound > maxxyz.z)) { break; }

        // Calculate how many 'dx,dy,dz' steps to take until hitting the next boundary.
        if (xdir == 0) { steps_to_take_x = 1.f; } else { steps_to_take_x = (nextxbound - lis.x)/lis.dx; }
        if (ydir == 0) { steps_to_take_y = 1.f; } else { steps_to_take_y = (nextybound - lis.y)/lis.dy; }
        if (zdir == 0) { steps_to_take_z = 1.f; } else { steps_to_take_z = (nextzbound - lis.z)/lis.dz; }
        // It's important to SNAP the new lis.x, lis.y, lis.z position to EXACTLY the nextxbound, nextybound or nextzbound! That's the reason for this somewhat convoluted approach.
        if (steps_to_take_x <= steps_to_take_y) {
            if (steps_to_take_x <= steps_to_take_z) {
                // Snap to next x
                steps_to_take = min(steps_to_take_x, steps_to_end);
                lis.x = nextxbound;
                lis.y += steps_to_take * lis.dy;
                lis.z += steps_to_take * lis.dz;
            } else {
                // Snap to next z
                steps_to_take = min(steps_to_take_z, steps_to_end);
                lis.x += steps_to_take * lis.dx;
                lis.y += steps_to_take * lis.dy;
                lis.z = nextzbound;
            }
        } else if (steps_to_take_y <= steps_to_take_z) {
            // Snap to next y
            steps_to_take = min(steps_to_take_y, steps_to_end);
            lis.x += steps_to_take * lis.dx;
            lis.y = nextybound;
            lis.z += steps_to_take * lis.dz;
        } else {
            // Snap to next z
            steps_to_take = min(steps_to_take_z, steps_to_end);
            lis.x += steps_to_take * lis.dx;
            lis.y += steps_to_take * lis.dy;
            lis.z = nextzbound;
        }

        idx_x = ((int)nextxbound - ((1+xdir)/2)); // x index in surface
        idx_y = ((int)nextybound - ((1+ydir)/2)); // y index in surface
        idx_z = ((int)nextzbound - ((1+zdir)/2)); // z index in surface

        if ( // Don't integrate the volume outside the bounds of the chunk!
            (idx_x >= texchunkspec.offsets.x) && (idx_x < texchunkspec.offsets.x + texchunkspec.shape.x) &&
            (idx_y >= texchunkspec.offsets.y) && (idx_y < texchunkspec.offsets.y + texchunkspec.shape.y) &&
            (idx_z >= texchunkspec.offsets.z) && (idx_z < texchunkspec.offsets.z + texchunkspec.shape.z)
        ) {
            if (!return_false_steps) {
                if (steps_to_take < 0) {
                    printf("Negative value encountered! %.3f\n", steps_to_take);
                }
                accum_val += steps_to_take * tex3D<Tv>(
                    volume_texture,
                    1.f/2.f + idx_z,  // z index in texture
                    1.f/2.f + idx_y,  // y index in texture
                    1.f/2.f + idx_x   // x index in surface
                );
            }
        } else {
            if (return_false_steps) {
                accum_val += 1.;
            }
        }

        steps_to_end -= steps_to_take; // If we've reached the end, then steps_to_end should be exactly 0.f, and the loop should stop
    }

    if (!return_false_steps) {
        accum_val *= length_per_step;
    }
    return accum_val;
}

__device__ fvec3 detpos(
        det_spec        dspec,
        unsigned int    det_idx_x,
        unsigned int    det_idx_y
)
{
    fvec3 dpos;
    if (dspec.type != 0) {
        // Not implemented!
    } else if (dspec.type == 0) {
        // Flat panel detector
        float xnum = ((float)det_idx_x - (dspec.nu-1)/2.);
        float ynum = ((float)det_idx_y - (dspec.nv-1)/2.);
        dpos = dspec.pos + (dspec.u*xnum) + (dspec.v*ynum);
    }
    return dpos;
}


template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp>  // type of projection data. e.g. float or double
__global__ void project_CB_texture_chunked(
        const config_CB_project   config,          // any miscellaneous config for the CB projection
        const proj_spec*          projspecs,       // contains the source and detector configurations
        const unsigned int        num_projs,       // length of projspecs array
        const vol_tex_chunk_spec  texchunkspec,
        cudaTextureObject_t       volume_texture,
        const uivec3              output_projs_strides,
        Tp*                       output_projs
)
{

    /*
    const unsigned long threadid = (
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y
    );
    const unsigned long vol_chunk_size = volchunk_Xsize*volchunk_Ysize*volchunk_Zsize;
    */
    unsigned int det_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int det_idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int proj_idx  = threadIdx.z + blockIdx.z * blockDim.z;

    // Now, if out of range of the detector, then return.
    if (proj_idx >= num_projs) {
        return;
    }
    proj_spec projspec = projspecs[proj_idx];
    if ((det_idx_x >= projspec.dspec.nu) ||
        (det_idx_y >= projspec.dspec.nv)) {
        return;
    }

    // Get the index 'det_idx' of the detector data that this thread should write to
    unsigned long det_idx = (
        output_projs_strides.y * det_idx_x +
        output_projs_strides.z * det_idx_y +
        output_projs_strides.x * proj_idx
    )/sizeof(Tp);

    // Get the start and end points of the projection line
    fvec3 source_pos = projspec.sourcepos;
    fvec3 det_pos    = detpos(projspec.dspec, det_idx_x, det_idx_y);

    // Calculate projection
    // ...
    Tp val = 0;
    val = exact_line_integral<Tv, Tp>(
            source_pos, det_pos, texchunkspec, volume_texture);
    // Do things ...
    //val = 18./norm_fvec3(det_pos - projspec.sourcepos);

    // Add the result atomically.
    // (using atomicAdd allows us to launch other kernels concurrently, if desired)
    atomicAdd(&output_projs[det_idx], val);
}
