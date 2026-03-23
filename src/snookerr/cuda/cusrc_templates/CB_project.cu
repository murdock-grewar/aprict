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
    /*
    // Check the rectangular bounds of the volume chunk.
    fvec3 volminbounds = vchunk_minbounds(vchunk_spec, vol_corner_pos);
    fvec3 volmaxbounds = vchunk_maxbounds(vchunk_spec, vol_corner_pos);
    */
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
__global__ void doathing(
        config_CB_project   config,          // any miscellaneous config for the CB projection
        //volume_chunk_spec   vchunk_spec,     // specifies the indices of the volume chunk.
        proj_spec           projspec,        // contains the source and detector configurations
        fvec3               vol_corner_pos,  // position of the 0,0,0 voxel of the volume
        uivec3              vol_strides,     // strides in memory (in bytes) of the x,y,z indices
        Tv*                 volume,
        uivec3              output_projs_strides,
        Tp*                 output_projs
)
{
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
    unsigned long threadid = (
        threadIdx.x +
        threadIdx.y * blockDim.x //+
        //threadIdx.z * blockDim.x * blockDim.y // It is assumed that blockDim.z == 1, i.e. threadIdx.z == 0.
    );
    // Calculate the number of volume elemnts that each thread must load
    unsigned long chunk_size = volchunk_Xsize*volchunk_Ysize*volchunk_Zsize;//vchunk_size(vchunk_spec);
    unsigned short num_volels_per_thread = ceil(chunk_size / (float)(blockDim.x * blockDim.y));
    // No reason to calculate these
    /*
    unsigned long blockid = (
        blockIdx.x +
        blockIdx.y * gridDim.x //+
        //blockIdx.z * gridDim.x * gridDim.y
    );
    */
    // Calculate the pixel on the detector corresponding to this thread
    unsigned int det_idx_x = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int det_idx_y = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("val: %.2f\n", val);

    // Populate the shared memory with the volume chunk
    for (int i = num_volels_per_thread*threadid; i < num_volels_per_thread*(threadid+1); i++) {
        if (i >= chunk_size) { break; }
        // Note the integer divisions are intentional.
        unsigned int j = i;
        unsigned int volidx_z = j/(volchunk_Xsize*volchunk_Ysize);
        j -= volidx_z*volchunk_Xsize*volchunk_Ysize;
        unsigned int volidx_y = j/volchunk_Xsize;
        j -= volidx_y*volchunk_Xsize;
        unsigned int volidx_x = j;
        // Load the volume data into the shared memory
        unsigned long mem_idx = (
            vol_strides.x * (volidx_x) +
            vol_strides.y * (volidx_y) +
            vol_strides.z * (volidx_z)
        )/sizeof(Tv);
        vchunk_shm[volidx_x][volidx_y][volidx_z] = volume[mem_idx];
    }
    __syncthreads(); // Let all the threads add to shared memory before continuing

    // Now, if out of range of the detector, then return
    if ((det_idx_x >= projspec.dspec.nu) || (det_idx_y >= projspec.dspec.nv)) {
        return;
    }

    // Do things ...
    fvec3 det_pos = detpos(projspec.dspec, det_idx_x, det_idx_y);
    unsigned long det_idx = (
        output_projs_strides.y * det_idx_x +
        output_projs_strides.z * det_idx_y
    )/sizeof(Tp);
    float val = 18./norm_fvec3(det_pos - projspec.sourcepos);
    output_projs[det_idx] = val;


    return;

    //unsigned int det_idx_z = threadIdx.z + blockIdx.z * blockDim.z;
    if (det_idx_y > 0) { return; }
    //atomicAdd(output_val, (Tp)det_idx_x);
    //output_val[0] = (Tp)(vchunk_spec.idx_xstop + vchunk_spec.idx_ystop);
    //output_val[0] = (Tp)num_volels_per_thread;
    if (det_idx_x > 0) { return; }
    Tv accum = 0;
    for (int i = 0; i < volchunk_Xsize; i++) {
        for (int j = 0; j < volchunk_Ysize; j++) {
            for (int k = 0; k < volchunk_Zsize; k++) {
                accum += vchunk_shm[i][j][k];
            }
        }
    }
    output_projs[0] = accum;
    output_projs[0] = 1.;
}
