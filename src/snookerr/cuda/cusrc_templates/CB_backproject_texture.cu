/*
 * In order to mantain O(N^3) scaling, we need a finite, closed-form solution
 * which maps a voxel to its projection onto the detector, in terms of the detector indices.
 *
 * (The alterantive is to run a thread for each detector index, but the lack of atomic addition
 * for surfaces/textures leads to the need to use a lot of global memory atomicAdds for large
 * volumes, which is unacceptably slow. Maybe it's possible to make the memory accesses more
 * efficient for that case; I'm not sure.)
 */


#include <cuda/atomic>

#include "base.h"
#include "geometry.h"
#include "config_cb_project.h"

template <typename T> float sgn(const T val) {
    return (T(0) < val) - (val < T(0));
}

template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp>  // type of projection data. e.g. float or double
__device__ Tv exact_line_integral_backproject(
        const Tp            value,
        const fvec3         v1,
        const fvec3         v2,
        const fvec3         voxpos // position of the center of the voxel
    )
{
    // Coordinates will be measured relative to the source point, v1
    const fvec3 relvoxpos = voxpos - v1;
    const fvec3 dv     = v2 - v1;
    // Flip coordinate as necessary so that dv has positive components
    const fvec3 sgnvec = {sgn(dv.x), sgn(dv.y), sgn(dv.z)};
    const fvec3 dv2    = sgnvec * dv;
    const fvec3 maxxyz2 = sgnvec * relvoxpos + fvec3(1./2., 1./2., 1./2.);
    // Maybe insert return here if any component of maxxyz2 is nonpositive...
    const fvec3 minxyz2 = maxxyz2         - fvec3(1., 1., 1.);
    const fvec3 minstepxyz = minxyz2 / dv2;
    const fvec3 maxstepxyz = maxxyz2 / dv2;
    const float a1 = max(0.f, max_of_components(minstepxyz));
    const float a2 = max(a1,  min_of_components(maxstepxyz));
    const float c = a2 - a1;
    return c * value * norm_fvec3(dv); // Maybe add an extra quick return if c is zero? Maybe compiler does it automatically?
}


template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp>  // type of projection data. e.g. float or double
__device__ void exact_line_integral_backproject_fast(
        const Tp            value,
        const fvec3         dv_positive_unit_vector,
        const fvec3         voxpos, // position of the center of the voxel
        Tv*                 accumulator
    )
{
    const fvec3 maxxyz2 = relvoxpos + fvec3(1./2., 1./2., 1./2.);
    const fvec3 minxyz2 = maxxyz2   - fvec3(1.,    1.,    1.);
    const fvec3 maxstepxyz = maxxyz2 / dv2;
    const float a1 = min(0.f, max_of_components(minstepxyz));
    const float a2 = min(a1,  min_of_components(maxstepxyz));
    const float c = a2 - a1;
    (*accumulator) += c * value; // Maybe add an extra quick return if c is zero? Maybe compiler does it automatically?
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
__global__ void backproject_CB_texture(
        const config_CB_project   config,          // any miscellaneous config for the CB projection
        //volume_chunk_spec   vchunk_spec,   // specifies the indices of the volume chunk.
        const proj_spec*          projspecs,       // contains the source and detector configurations
        const unsigned int        num_projs,       // length of projspecs array
        const vol_tex_chunk_spec  texchunkspec,
        cudaSurfaceObject_t       volume_surface,
        const uivec3              input_projs_strides,
        const Tp*                 input_projs
)
{
    // Get the position within the volume surface
    const uivec3 vol_surf_idx = texchunkspec.offsets + uivec3(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    );

    // Return if block is out of range
    if ((vol_surf_idx.x >= texchunkspec.offsets.x + texchunkspec.shape.x) ||
        (vol_surf_idx.y >= texchunkspec.offsets.y + texchunkspec.shape.y) ||
        (vol_surf_idx.z >= texchunkspec.offsets.z + texchunkspec.shape.z)) {
        return;
    }


    // Get the center position of this voxel
    const fvec3 vox_pos = texchunkspec.texpos + vol_surf_idx + fvec3(1./2., 1./2., 1./2.);

    // Reused structures
    proj_spec projspec;
    fvec3     source_pos;
    fvec3     det_pos;
    unsigned long det_idx;
    Tp        det_value;

    // Loop over all detector pixels and add the backprojection
    Tv accumulator = 0.;
    for (unsigned int proj_idx = 0; proj_idx < num_projs; proj_idx++) {
        // Get the start point of the projection line
        projspec   = projspecs[proj_idx];
        source_pos = projspec.sourcepos;

        // Find the intersection with the detector of the ray passing through
        // the center of the voxel
        fvec2 det_coords_c = get_projected_detector_coords(vox_pos, projspec);
        /*
        fvec3 det_pos_c    = projspec.dspec.pos + (
            projspec.dspec.u * det_coords_c.x +
            projspec.dspec.v * det_coords_c.y
        );
        */

        // Find the intersection with the detector of the ray passing through
        // each of the eight corners of the voxel. These will define the range
        // of pixels that are in range and need to be considered for backprojection.
        fvec3 vox_pos_offset;
        fvec2 det_coords;
        fvec2 det_coords_min = det_coords_c;
        fvec2 det_coords_max = det_coords_c;
        fvec3 offsets[8] = {
            fvec3(-1./2., -1./2., -1./2.),
            fvec3(-1./2., -1./2.,  1./2.),
            fvec3(-1./2.,  1./2., -1./2.),
            fvec3(-1./2.,  1./2.,  1./2.),
            fvec3( 1./2., -1./2., -1./2.),
            fvec3( 1./2., -1./2.,  1./2.),
            fvec3( 1./2.,  1./2., -1./2.),
            fvec3( 1./2.,  1./2.,  1./2.)
        };
        for (const fvec3 &offset : offsets) {
            vox_pos_offset = vox_pos + offset;
            det_coords = get_projected_detector_coords(vox_pos_offset, projspec);
            det_coords_min = {min(det_coords_min.x, det_coords.x), min(det_coords_min.y, det_coords.y)};
            det_coords_max = {max(det_coords_max.x, det_coords.x), max(det_coords_max.y, det_coords.y)};
        }
        int det_idx_x_min = max(0.f, floor(det_coords_min.x + projspec.dspec.nu/2.f));
        int det_idx_y_min = max(0.f, floor(det_coords_min.y + projspec.dspec.nv/2.f));
        int det_idx_x_max = min(projspec.dspec.nu-1.f, ceil(det_coords_max.x + projspec.dspec.nu/2.f));
        int det_idx_y_max = min(projspec.dspec.nv-1.f, ceil(det_coords_max.y + projspec.dspec.nv/2.f));
        for (int det_idx_x = det_idx_x_min; det_idx_x < det_idx_x_max+1; det_idx_x++) {
            for (int det_idx_y = det_idx_y_min; det_idx_y < det_idx_y_max+1; det_idx_y++) {

                fvec3 det_pos      = projspec.dspec.pos + (
                    projspec.dspec.u * (det_idx_x - (projspec.dspec.nu-1)/2.) +
                    projspec.dspec.v * (det_idx_y - (projspec.dspec.nv-1)/2.)
                );

                // Find the detector value there
                //
                det_idx = (
                    input_projs_strides.y * det_idx_x +
                    input_projs_strides.z * det_idx_y +
                    input_projs_strides.x * proj_idx
                )/sizeof(Tp);
                det_value = input_projs[det_idx];

                // Compute the backprojection into this voxel
                accumulator += exact_line_integral_backproject<Tv, Tp>(
                    det_value, source_pos, det_pos, vox_pos);

            }
        }
    }

    // Write the value to the surface memory and leave
    if (accumulator == 0) { return; }
    Tv old_val = surf3Dread<Tv>(
        volume_surface,
        vol_surf_idx.z * sizeof(Tv),
        vol_surf_idx.y,
        vol_surf_idx.x
    );
    surf3Dwrite(
        old_val + accumulator,
        volume_surface,
        vol_surf_idx.z * sizeof(Tv),
        vol_surf_idx.y,
        vol_surf_idx.x
    );
}
