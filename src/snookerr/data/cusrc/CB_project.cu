#include <cuda/atomic>

// Provides config_CB_project, volume_chunk_spec, det_spec, fvec3
#include "config_cb_project.h"

__device__ fvec3 vchunk_idx_to_pos(
        fvec3 idx_f,
        volume_chunk_spec   vchunk_spec,
        fvec3               vol_corner_pos)
{
    fvec3 pos;
    pos.x = vol_corner_pos.x + idx_f.x + (float)vchunk_spec.idx_xstart;
    pos.y = vol_corner_pos.y + idx_f.y + (float)vchunk_spec.idx_ystart;
    pos.z = vol_corner_pos.z + idx_f.z + (float)vchunk_spec.idx_zstart;
    return pos;
}


__device__ fvec3 vchunk_minbounds(
        volume_chunk_spec   vchunk_spec,
        fvec3               vol_corner_pos)
{
    fvec3 idx_f;
    idx_f.x = -(1./2) + vchunk_spec.idx_xstart;
    idx_f.y = -(1./2) + vchunk_spec.idx_ystart;
    idx_f.z = -(1./2) + vchunk_spec.idx_zstart;
    return vchunk_idx_to_pos(idx_f, vchunk_spec, vol_corner_pos);
}


__device__ fvec3 vchunk_maxbounds(
        volume_chunk_spec   vchunk_spec,
        fvec3               vol_corner_pos)
{
    fvec3 idx_f;
    idx_f.x = -(1./2) + vchunk_spec.idx_xstop;
    idx_f.y = -(1./2) + vchunk_spec.idx_ystop;
    idx_f.z = -(1./2) + vchunk_spec.idx_zstop;
    return vchunk_idx_to_pos(idx_f, vchunk_spec, vol_corner_pos);
}


__device__ fvec3 vchunk_center_idx_f( // float index of the center of the chunk
        volume_chunk_spec   vchunk_spec,
        fvec3               vol_corner_pos
)
{
    fvec3 idx_f;
    idx_f.x = ((float)(vchunk_spec.idx_xstart + vchunk_spec.idx_xstop - 1))/2.f;
    idx_f.x = ((float)(vchunk_spec.idx_xstart + vchunk_spec.idx_xstop - 1))/2.f;
    idx_f.x = ((float)(vchunk_spec.idx_xstart + vchunk_spec.idx_xstop - 1))/2.f;
    return idx_f;
}


__device__ fvec3 vchunk_center_pos( // float position of the center of the chunk
        volume_chunk_spec   vchunk_spec,
        fvec3               vol_corner_pos
)
{
    return vchunk_idx_to_pos(vchunk_center_idx_f(vchunk_spec, vol_corner_pos), vchunk_spec, vol_corner_pos);
}


__device__ unsigned long vchunk_size(
        volume_chunk_spec   vchunk_spec // specifies the indices of the volume chunk.
)
{
    return (
        (unsigned long)(vchunk_spec.idx_xstop - vchunk_spec.idx_xstart) *
        (unsigned long)(vchunk_spec.idx_ystop - vchunk_spec.idx_ystart) *
        (unsigned long)(vchunk_spec.idx_zstop - vchunk_spec.idx_zstart)
    );
}

/*
template<typename Tv, typename Tp> // These types are intended to be floating point.
__device__ Tp project_vchunk(
        config_CB_project   config,
        volume_chunk_spec   vchunk_spec, // specifies the indices of the volume chunk.
        fvec3               sourcepos,
        fvec3               det_pos,
        fvec3               vol_corner_pos, // position of the 0,0,0 voxel of the volume
        // Shared memory containing the volume chunk data.
        //__shared__ cuda::atomic<Tv, cuda::thread_scope_block>* vchunk_shm)
        Tv***               vchunk_shm)
{
    // Abort if the projection line is entirely outside the rectangular bounds.
    // HOTFIX: for now, I will just check if the line is outside the circumscribed sphere

    // Check the rectangular bounds of the volume chunk.
    //fvec3 volminbounds = vchunk_minbounds(vchunk_spec, vol_corner_pos);
    //fvec3 volmaxbounds = vchunk_maxbounds(vchunk_spec, vol_corner_pos);

    fvec3 chunk_center = vchunk_center_pos(vchunk_spec, vol_corner_pos);
    fvec3 delta_pos = det_pos - sourcepos;
    delta_pos = delta_pos / norm_fvec3(delta_pos);
    fvec3 nearest_point = sourcepos - (delta_pos * dot_prod(sourcepos - chunk_center, delta_pos));
    float maxrad = max(
            vchunk_spec.idx_xstop - volchunk_spec.idx_xstart,
            vchunk_spec.idx_ystop - volchunk_spec.idx_ystart,
            vchunk_spec.idx_zstop - volchunk_spec.idx_zstart
    );
    if (norm_fvec3(nearest_point - chunk_center) > maxrad) {
        return (Tp)0.;
    }
    return (Tp)1.;

}
*/


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
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp,  // type of projection data. e.g. float or double
    unsigned int volchunk_Xsize,  // volume chunk size in X dimension
    unsigned int volchunk_Ysize,  // volume chunk size in Y dimension
    unsigned int volchunk_Zsize>  // volume chunk size in Z dimension
__device__ Tp exact_line_integral(
        fvec3       v1,
        fvec3       v2,
        vol_spec    volspec,
        Tv          (*vchunk_shm)[volchunk_Xsize][volchunk_Ysize][volchunk_Zsize]  // These values will be used instead of the volume from 'volspec'.
)
{
    // MODE OF OPERATION
    // This function steps between boundaries of voxels, and adds the exact line integral components.

    // Measure distances relative to the bottom-left corner of the 000 voxel.
    // This way, integer values for x,y,z correspond to voxel boundaries.
    float xstart = v1.x - volspec.pos_000.x + (1./2.);
    float ystart = v1.y - volspec.pos_000.y + (1./2.);
    float zstart = v1.z - volspec.pos_000.z + (1./2.);
    float xend   = v2.x - volspec.pos_000.x + (1./2.);
    float yend   = v2.y - volspec.pos_000.y + (1./2.);
    float zend   = v2.z - volspec.pos_000.z + (1./2.);

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
    while (steps_to_end > 0) { // Loop until 'steps_to_end' becomes nonpositive, which would indicate that we have reached the end of the line integral
        // Find the critical values of x, y, z
        // Initially, I had planned to calculate this once at the beginning, and only update the necessary x/y/z values as necessary.
        // However, that approach would miss corner cases where the ray hits two boundaries at once. It's safer to totally recalculate these each loop.
        if (xdir <= 0) { nextxbound = ceil(lis.x) - 1; } else { nextxbound = floor(lis.x) + 1; }
        if (ydir <= 0) { nextybound = ceil(lis.y) - 1; } else { nextybound = floor(lis.y) + 1; }
        if (zdir <= 0) { nextzbound = ceil(lis.z) - 1; } else { nextzbound = floor(lis.z) + 1; }
        if ((xdir <= 0) && (nextxbound < 0))              { break; }
        if ((xdir >  0) && (nextxbound > volchunk_Xsize)) { break; }
        if ((ydir <= 0) && (nextybound < 0))              { break; }
        if ((ydir >  0) && (nextybound > volchunk_Ysize)) { break; }
        if ((zdir <= 0) && (nextzbound < 0))              { break; }
        if ((zdir >  0) && (nextzbound > volchunk_Zsize)) { break; }

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
            (idx_x >= 0) && (idx_x < volchunk_Xsize) &&
            (idx_y >= 0) && (idx_y < volchunk_Ysize) &&
            (idx_z >= 0) && (idx_z < volchunk_Zsize)
        ) {
            accum_val += steps_to_take * (*vchunk_shm)[idx_x][idx_y][idx_z];
        }

        steps_to_end -= steps_to_take; // If we've reached the end, then steps_to_end should be exactly 0.f, and the loop should stop
    }

    accum_val *= length_per_step;
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
        dpos = dspec.pos + dspec.u*xnum + dspec.v*ynum;
    }
    return dpos;
}


template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp,  // type of projection data. e.g. float or double
    unsigned int volchunk_Xsize,  // volume chunk size in X dimension
    unsigned int volchunk_Ysize,  // volume chunk size in Y dimension
    unsigned int volchunk_Zsize>  // volume chunk size in Z dimension
__global__ void project_CB_sharedmem(
        const config_CB_project   config,          // any miscellaneous config for the CB projection
        //volume_chunk_spec   vchunk_spec,   // specifies the indices of the volume chunk.
        const proj_spec*          projspecs,       // contains the source and detector configurations
        const unsigned int        num_projs,       // length of projspecs array
        const vol_spec            volspec,
        const uivec3              output_projs_strides,
        Tp*                       output_projs
)
{

    const uivec3 vol_strides    = volspec.strides_bytes;
    const Tv*    volume         = (Tv*)volspec.ptr_000;
    // This kernel should be used with a blockDim.z of 1.
    // i.e. only use blocks with a X and Y width. These blocks will be tiled on the detector
    // pixels, and each thread will write to one detector pixel.

    //// Load the relevant part of the volume into memory
    // The work of loading the volume chunk into shared memory will be shared between the
    // threads. It is most efficient to have adjacent threads load adjacent pieces of memory,
    // so for this reason, we care about the actual thread id, and not just the x/y position
    // on the detector.
    //
    // It will be assumed that the volume has memory strides that are smallest in x, then y,
    // then the largest strides in z. #TODO: generalise this.
    __shared__ Tv vchunk_shm[volchunk_Xsize][volchunk_Ysize][volchunk_Zsize];
    const unsigned long threadid = (
        threadIdx.x +
        threadIdx.y * blockDim.x +
        threadIdx.z * blockDim.x * blockDim.y
    );
    // Calculate the number of volume elemnts that each thread must load
    const unsigned long vol_chunk_size = volchunk_Xsize*volchunk_Ysize*volchunk_Zsize;//vchunk_size(vchunk_spec);
    const unsigned short num_volels_per_thread = ceil(vol_chunk_size / (float)(blockDim.x * blockDim.y * blockDim.z));
    // No reason to calculate these
    /*
    unsigned long blockid = (
        blockIdx.x +
        blockIdx.y * gridDim.x //+
        //blockIdx.z * gridDim.x * gridDim.y
    );
    */
    // Calculate the pixel on the detector corresponding to this thread
    // Note: so that thread warps can return early, I want the projection index (proj_idx)
    // to correspond to the largest stride in thread number. Hence, it is associated with Z.
    unsigned int det_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int det_idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int proj_idx  = threadIdx.z + blockIdx.z * blockDim.z;

    // Populate the shared memory with the volume chunk
    for (int i = num_volels_per_thread*threadid; i < num_volels_per_thread*(threadid+1); i++) {
        if (i >= vol_chunk_size) { break; }
        // Note the integer divisions are intentional.
        unsigned int j = i;
        unsigned int volidx_z = j/(volchunk_Xsize*volchunk_Ysize);
        j -= volidx_z*volchunk_Xsize*volchunk_Ysize;
        unsigned int volidx_y = j/volchunk_Xsize;
        j -= volidx_y*volchunk_Xsize;
        unsigned int volidx_x = j;
        if (
            (volidx_x < volspec.shape.x) &&
            (volidx_y < volspec.shape.y) &&
            (volidx_z < volspec.shape.z)
        ) {
            // Load the volume data into the shared memory
            unsigned long mem_idx = (
                vol_strides.x * (volidx_x) +
                vol_strides.y * (volidx_y) +
                vol_strides.z * (volidx_z)
            )/sizeof(Tv);
            vchunk_shm[volidx_x][volidx_y][volidx_z] = volume[mem_idx];
        } else {
            // The position is out of bounds of the volumue. Set it to 0 in the shared memory.
            vchunk_shm[volidx_x][volidx_y][volidx_z] = 0.;
        }
    }
    __syncthreads(); // Let all the threads add to shared memory before continuing

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
    val = exact_line_integral<Tv, Tp, volchunk_Xsize, volchunk_Ysize, volchunk_Zsize>(
            source_pos, det_pos, volspec, &vchunk_shm);
    // Do things ...
    //val = 18./norm_fvec3(det_pos - projspec.sourcepos);

    // Add the result atomically.
    // (using atomicAdd allows us to launch other kernels concurrently, if desired)
    atomicAdd(output_projs + det_idx, val);
}
