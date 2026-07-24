/*
 * In order to mantain O(N^3) scaling, we need a finite, closed-form solution
 * which maps a voxel to its projection onto the detector, in terms of the detector indices.
 *
 * (The alterantive is to run a thread for each detector index, but the lack of atomic addition
 * for surfaces/textures leads to the need to use a lot of global memory atomicAdds for large
 * volumes, which is unacceptably slow. Maybe it's possible to make the memory accesses more
 * efficient for that case; I'm not sure.)
 */

// #include <cuda/std/type_traits>
#include <cuda/atomic>
#include <cuda_fp16.h>

#include "base.h"
#include "geometry.h"
#include "config_cb_backproject.h"

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


template <typename T> float sgn(const T val) {
    return (T(0) < val) - (val < T(0));
}

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
    uivec3                  idx
)
{ surf3Dwrite(value, surface,  idx.z * sizeof(T), idx.y, idx.x); }
template<>
__device__ void write_to_surf3D<half>(
    half                    value,
    cudaSurfaceObject_t     surface,
    uivec3                  idx
)
{ write_to_surf3D<unsigned short>(__half_as_ushort(value), surface, idx); }
template<typename T>
__device__ T read_from_surf3D(
    cudaSurfaceObject_t     surface,
    uivec3                  idx
)
{ return surf3Dread<T>(surface, idx.z * sizeof(T), idx.y, idx.x); }
template<>
__device__ half read_from_surf3D<half>(
    cudaSurfaceObject_t     surface,
    uivec3                  idx
)
{ return __ushort_as_half(read_from_surf3D<unsigned short>(surface, idx)); }


__device__ float exact_line_integral_backproject_linelen_unitvoxwid(
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
    return c * norm_fvec3(dv); // Maybe add an extra quick return if c is zero? Maybe compiler does it automatically?
}

__device__ float exact_line_integral_backproject_linelen(
        const fvec3         v1,
        const fvec3         v2,
        const fvec3         voxpos, // position of the center of the voxel
        const float         voxwid
    )
{
    return voxwid * exact_line_integral_backproject_linelen_unitvoxwid(
        v1 / voxwid,
        v2 / voxwid,
        voxpos / voxwid
    );
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

template<typename Tp>
__device__ Tp get_data_2d_guarded(
        const int2  idx,
        const uint2 shape,
        const ulong2 strides,
        const Tp*   data_2d
)
{
    if ((idx.x < 0) || (idx.x >= shape.x)) { return (Tp)0; }
    if ((idx.y < 0) || (idx.y >= shape.y)) { return (Tp)0; }
    // Find the detector value there
    const unsigned long arr_idx = (
        strides.x * idx.x +
        strides.y * idx.y
    )/sizeof(Tp);
    const Tp value = data_2d[arr_idx];
    return value;
}

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

template<typename Tp>
__device__ Tp get_bilinear_interp(
        const float2 det_idx_f,
        const uint2  shape,
        const ulong2  strides,
        const Tp*    data_2d
)
{
    // Will assume zeroes outside data_2d
    // det_idx_f = (0,0) means the CENTER of the bottom-left pixel

    // Guard against reading totally outside texture
    if (
        (det_idx_f.x <= -1) ||
        (det_idx_f.y <= -1) ||
        (det_idx_f.x >= shape.x) ||
        (det_idx_f.y >= shape.y)
    ) {
        return (Tp)0;
    }

    int2 idx_lower = int2({(int)floor(det_idx_f.x), (int)floor(det_idx_f.y)});
    int2 idx_upper = int2({idx_lower.x + 1, idx_lower.y + 1});
    float2 fractional_remainders = float2({det_idx_f.x - idx_lower.x, det_idx_f.y - idx_lower.y});

    Tp val_00 = get_data_2d_guarded<Tp>(idx_lower, shape, strides, data_2d);
    Tp val_01 = get_data_2d_guarded<Tp>(int2({idx_lower.x, idx_upper.y}), shape, strides, data_2d);
    Tp val_10 = get_data_2d_guarded<Tp>(int2({idx_upper.x, idx_lower.y}), shape, strides, data_2d);
    Tp val_11 = get_data_2d_guarded<Tp>(idx_upper, shape, strides, data_2d);

    Tp tmp1 = val_00 * fcast<Tp>(1.f - fractional_remainders.x) + val_10 * fcast<Tp>(fractional_remainders.x);
    Tp tmp2 = val_01 * fcast<Tp>(1.f - fractional_remainders.x) + val_11 * fcast<Tp>(fractional_remainders.x);
    return tmp1 * fcast<Tp>(1.f - fractional_remainders.y) + tmp2 * fcast<Tp>(fractional_remainders.y);
}

template<typename Tp>
__device__ Tp get_zero_interp(
        const float2 det_idx_f,  // measured from the centre of the corner pixel, in multiples of pixel-lengths
        const uint2  shape,
        const ulong2 strides,
        const Tp*    data_2d
)
{
    const int2 det_idx = int2({
        (int)floor(det_idx_f.x + 0.5f),
        (int)floor(det_idx_f.y + 0.5f)
    });

    return get_data_2d_guarded<Tp>(det_idx, shape, strides, data_2d);

}

template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp>  // type of projection data. e.g. float or double
__device__ Tv _backproject_CB_getval(
        const config_CB_backproject     config,          // any miscellaneous config for the CB projection
        const fvec3                     pos_000_c,       // absolute x,y,z position of the center of the voxel with index 0,0,0 within the chunk
        const uivec3                    vol_chunk_shape,
        const proj_chunk_spec_maxlen10  proj_chunk_spec,
        const uivec3                    vol_idx,          // index of the voxel within the chunk, beginning from 0,0,0
        const float                     voxwid
)
{

    // Return 0 if block is out of range
    if ((vol_idx.x >= vol_chunk_shape.x) ||
        (vol_idx.y >= vol_chunk_shape.y) ||
        (vol_idx.z >= vol_chunk_shape.z)) {
        return 0.;
    }

    // Get the center position of this voxel
    const fvec3 vox_pos = pos_000_c + (vol_idx * voxwid);

    // Skip if masking to the cylinder, and out of range
    if (config.mask_to_cylinder) {
        if ((vox_pos.x*vox_pos.x + vox_pos.y*vox_pos.y) > config.mask_to_cyl_rad*config.mask_to_cyl_rad) {
            return 0;
        }
    }

    // Unpack projection metadata
    const ulvec3    input_projs_strides = proj_chunk_spec.projmemstrides;
    const Tp*       input_projs         = (Tp*)proj_chunk_spec.projmemaddr;
    const unsigned short num_projs      = proj_chunk_spec.numprojs;

    // Reused structures
    fvec3     source_pos;
    unsigned long det_idx;

    const bool exact = config.exact_line_integ_backproj;
    const Tv voxwid_Tv = fcast<Tv>(voxwid);

    // Loop over all detector pixels and add the backprojection
    const fvec3 _vox_corner_offsets[8] = {
        fvec3(-1./2., -1./2., -1./2.) * voxwid,
        fvec3(-1./2., -1./2.,  1./2.) * voxwid,
        fvec3(-1./2.,  1./2., -1./2.) * voxwid,
        fvec3(-1./2.,  1./2.,  1./2.) * voxwid,
        fvec3( 1./2., -1./2., -1./2.) * voxwid,
        fvec3( 1./2., -1./2.,  1./2.) * voxwid,
        fvec3( 1./2.,  1./2., -1./2.) * voxwid,
        fvec3( 1./2.,  1./2.,  1./2.) * voxwid
    };
    Tv accumulator = 0.;
    Tv sub_accumulator = 0.;
    for (unsigned int proj_idx = 0; proj_idx < num_projs; proj_idx++) {
        // Get the start point of the projection line
        //proj_spec projspec = proj_tex_chunk_spec.projspecs[proj_idx];
        proj_spec projspec = proj_chunk_spec.projspecs[proj_idx];
        source_pos = projspec.sourcepos;

        // Find the intersection with the detector of the ray passing through
        // the center of the voxel
        fvec2 det_coords_c = get_projected_detector_coords(vox_pos, projspec);

        sub_accumulator = (Tv)0;

        if (exact) {
            // Find the intersection with the detector of the ray passing through
            // each of the eight corners of the voxel. These will define the range
            // of pixels that are in range and need to be considered for backprojection.
            fvec3 vox_pos_offset;
            fvec2 det_coords;
            fvec2 det_coords_min = det_coords_c;
            fvec2 det_coords_max = det_coords_c;
            for (const fvec3 &offset : _vox_corner_offsets) {
                vox_pos_offset = vox_pos + offset;
                det_coords = get_projected_detector_coords(vox_pos_offset, projspec);
                det_coords_min = {min(det_coords_min.x, det_coords.x), min(det_coords_min.y, det_coords.y)};
                det_coords_max = {max(det_coords_max.x, det_coords.x), max(det_coords_max.y, det_coords.y)};
            }
            int det_idx_x_min = max(0, (int)floor(det_coords_min.x + projspec.dspec.nu/2.f));
            int det_idx_y_min = max(0, (int)floor(det_coords_min.y + projspec.dspec.nv/2.f));
            int det_idx_x_max = min(projspec.dspec.nu-1, (int)ceil(det_coords_max.x + projspec.dspec.nu/2.f));
            int det_idx_y_max = min(projspec.dspec.nv-1, (int)ceil(det_coords_max.y + projspec.dspec.nv/2.f));
            for (int det_idx_x = det_idx_x_min; det_idx_x <= det_idx_x_max; det_idx_x++) {
                for (int det_idx_y = det_idx_y_min; det_idx_y <= det_idx_y_max; det_idx_y++) {

                    fvec3 det_pos      = projspec.dspec.pos + (
                        projspec.dspec.u * (det_idx_x - (projspec.dspec.nu-1)/2.) +
                        projspec.dspec.v * (det_idx_y - (projspec.dspec.nv-1)/2.)
                    );

                    float intersect_linelen = exact_line_integral_backproject_linelen(
                        source_pos, det_pos, vox_pos, voxwid);

                    if (intersect_linelen != 0.f) {
                        // Find the detector value there
                        Tp det_value;
                        if (config.backproject_all_1s) {
                            det_value = 1.;
                        } else {
                            det_idx = (
                                input_projs_strides.y * det_idx_x +
                                input_projs_strides.z * det_idx_y +
                                input_projs_strides.x * proj_idx
                            )/sizeof(Tp);
                            det_value = input_projs[det_idx];
                        }

                        /*
                        float det_value = tex2DLayered<Tp>(
                            detector_texture,
                            det_idx_y + 1./2.f,
                            det_idx_x + 1./2.f,
                            proj_idx + tex_start_layer
                        );
                        */

                        // Compute the backprojection into this voxel
                        sub_accumulator += fcast<Tv>(fcast<float>(det_value) * intersect_linelen);
                    }

                }
            }
            // Apply Z_factor_power, if applicable
            if (config.Z_on_L_factor_power != 0) {
                fvec3 Lvec = projspec.dspec.pos - source_pos;
                float Z_on_L = dot_prod((vox_pos - source_pos), Lvec) / dot_prod(Lvec, Lvec);
                sub_accumulator *= fcast<Tv>(powint(Z_on_L, config.Z_on_L_factor_power));
            }
            if (config.Z_on_val_factor_power != 0) {
                float val = config.Z_on_val_factor_val;
                fvec3 Lvec = projspec.dspec.pos - source_pos;
                float Z_on_val = dot_prod((vox_pos - source_pos) / val, Lvec) / norm_fvec3(Lvec);
                sub_accumulator *= fcast<Tv>(powint(Z_on_val, config.Z_on_val_factor_power));
            }
        } else {
            // config.exact is FALSE

            // In the approximate method, the voxel fetches the value from the detector
            // by casting a ray. We use zeroth-order interpolation on the detector.

            /*
            // Zeroth-order interpolation
            // 3 voxels across: middle of detector is 0, which has index 1
            // 4 voxels across: middle of detector is 0, which has index (4-1)/2 = 1.5
            // below 1.5 should be rounded down to 1. above 1.5 should be rounded up to 2
            // So, we need to take floor(coord + N/2)
            int det_idx_x = floor(det_coords_c.x + (projspec.dspec.nu/2.));
            if ((det_idx_x < 0) || (det_idx_x >= projspec.dspec.nu)) { continue; }
            int det_idx_y = floor(det_coords_c.y + (projspec.dspec.nv/2.));
            if ((det_idx_y < 0) || (det_idx_y >= projspec.dspec.nv)) { continue; }
            // Find the detector value there
            det_idx = (
                input_projs_strides.y * det_idx_x +
                input_projs_strides.z * det_idx_y +
                input_projs_strides.x * proj_idx
            )/sizeof(Tp);
            Tp det_value = input_projs[det_idx];
            */

            // First-order interpolation (bilinear)
            float2 det_idx_f = float2({
                det_coords_c.x + ((projspec.dspec.nu-1)/2.f),
                det_coords_c.y + ((projspec.dspec.nv-1)/2.f)
            });
            uint2 shape   = uint2({projspec.dspec.nu, projspec.dspec.nv});
            ulong2 strides = ulong2({input_projs_strides.y, input_projs_strides.z});
            Tp det_value;
            if (config.approx_mode_linear_interpolate) {
                det_value = get_bilinear_interp(
                    det_idx_f, shape, strides, &input_projs[input_projs_strides.x * proj_idx / sizeof(Tp)]);
            } else {
                det_value = get_zero_interp(
                    det_idx_f, shape, strides, &input_projs[input_projs_strides.x * proj_idx / sizeof(Tp)]);
            }

            if (config.approx_mode_direct_raycast) {
                Tp multiplier = 1;
                if (
                    (config.Z_on_L_factor_power != 0) ||
                    (config.Z_on_val_factor_power != 0) ||
                    (config.backproj_weighting_preset != 0)
                ) {
                    fvec3 Lvec = projspec.dspec.pos - source_pos;
                    float Z = dot_prod((vox_pos - source_pos), Lvec) / norm_fvec3(Lvec);
                    if (config.Z_on_L_factor_power != 0) {
                        float Z_on_L = Z / norm_fvec3(Lvec);
                        multiplier *= fcast<Tp>(powint(Z_on_L, config.Z_on_L_factor_power));
                    }
                    if (config.Z_on_val_factor_power != 0) {
                        float val = config.Z_on_val_factor_val;
                        float Z_on_val = Z / val;
                        multiplier *= fcast<Tp>(powint(Z_on_val, config.Z_on_val_factor_power));
                    }
                    if (
                        (config.backproj_weighting_preset == 1) ||
                        (config.backproj_weighting_preset == 2)
                    ) {
                        // Assume that 'R' is contained in 'config.Z_on_val_factor_val'
                        float R = config.Z_on_val_factor_val;
                        fvec3 r_minus_s = vox_pos - source_pos;
                        float ell_on_R = norm_fvec3(fvec3(r_minus_s.x, r_minus_s.y, 0.f))/R;
                        float Z_on_R = Z / R;
                        float c_on_R = Z_on_R / ell_on_R;
                        float dsigned_on_R = c_on_R - ell_on_R;  // positive if the source point is further away
                        if (config.backproj_weighting_preset == 1) {
                            // multiplier /= fcast<Tp>(2*powint(c_on_R, 2)*(powint(c_on_R, 2) + powint(dsigned_on_R, 2)));
                            multiplier /= fcast<Tp>(powint(c_on_R, 2)*(powint(c_on_R, 2) + powint(dsigned_on_R, 2)));
                        }
                        if (config.backproj_weighting_preset == 2) {
                            float dsigned_on_c = dsigned_on_R / c_on_R;
                            float rho_on_R = sqrt(powint(vox_pos.x, 2) + powint(vox_pos.y, 2))/R;
                            float vyrel_on_R = (source_pos.x*vox_pos.y - source_pos.y*vox_pos.x)/R/R;
                            float vxrel_on_R = dot_prod(source_pos, vox_pos)/R/R;
                            multiplier *= fcast<Tp>(
                                -1 *
                                2 *
                                powint(1 - powint(rho_on_R, 2), 2) /
                                powint(powint(c_on_R, 2) + powint(dsigned_on_R, 2), 2) *
                                (-vyrel_on_R) * (
                                    1 + (
                                        vxrel_on_R / Z_on_R
                                    )
                                ) /
                                powint(ell_on_R, 2) /
                                R  // Need to divide by R because the formula takes a derivative with respect to u in the midplane coordinates, i.e. d/d(uR)
                            );
                        }
                    }
                }
                sub_accumulator = fcast<Tv>(det_value * multiplier);
            } else {
                // Compute the physical distance to the source
                fvec3 det_pos      = projspec.dspec.pos + (
                    projspec.dspec.u * det_coords_c.x +
                    projspec.dspec.v * det_coords_c.y
                );
                float dist_to_det = norm_fvec3(det_pos - source_pos);
                float dist_to_vox = norm_fvec3(vox_pos - source_pos);
                float mag = dist_to_det/dist_to_vox*voxwid;  // Need to divide by voxwid here
                float pixel_area = norm_fvec3(projspec.dspec.u) * norm_fvec3(projspec.dspec.v);
                Tp multiplier = fcast<Tp>(powint(mag,2) / pixel_area);

                if (config.Z_on_L_factor_power != 0) {
                    fvec3 Lvec = projspec.dspec.pos - source_pos;
                    float Z_on_L = dot_prod((vox_pos - source_pos), Lvec) / dot_prod(Lvec, Lvec);
                    multiplier *= fcast<Tp>(powint(Z_on_L, config.Z_on_L_factor_power));
                }
                if (config.Z_on_val_factor_power != 0) {
                    float val = config.Z_on_val_factor_val;
                    fvec3 Lvec = projspec.dspec.pos - source_pos;
                    float Z_on_val = dot_prod((vox_pos - source_pos) / val, Lvec) / norm_fvec3(Lvec);
                    multiplier *= fcast<Tp>(powint(Z_on_val, config.Z_on_val_factor_power));
                }
                sub_accumulator = voxwid_Tv * fcast<Tv>(det_value * multiplier);
            }

        }

        // Apply softening toward detector edges, if applicable
        const float sep = config.soften_edge_proportion;
        if ((sep > 0.f) && (sub_accumulator != (Tv)0)) {
            // Define distances from center, from 0 to 1
            // float udist = abs((det_coords_c.x - (projspec.dspec.nu-1)/2.)/(projspec.dspec.nu/2.f));
            // float vdist = abs((det_coords_c.y - (projspec.dspec.nv-1)/2.)/(projspec.dspec.nv/2.f));
            float udist = abs(det_coords_c.x)/(projspec.dspec.nu/2.f);
            float vdist = abs(det_coords_c.y)/(projspec.dspec.nv/2.f);
            // Scale each from 0 to 1, where 0 is completely softened (set to 0), and 1 is not softened at all
            udist = max(0.f, min(1.f, (1.f - udist)/sep));
            vdist = max(0.f, min(1.f, (1.f - vdist)/sep));
            // Set each to the corresponding softening factor
            udist = max(0.f, min(1.f, (1 + cos(3.1415926535898f * (1.f - udist)))/2.f));
            vdist = max(0.f, min(1.f, (1 + cos(3.1415926535898f * (1.f - vdist)))/2.f));
            // Apply the softening factors
            sub_accumulator *= udist;
            sub_accumulator *= vdist;
        }

        accumulator += sub_accumulator;
    }


    return accumulator;
}


// DEPRECATED!! May ignore some options, or behave in unexpected ways!
template<
    typename Tv,  // type of volume data. e.g. float or double.
    typename Tp>  // type of projection data. e.g. float or double
__global__ void backproject_CB_volmem(
        const config_CB_backproject     config,          // any miscellaneous config for the CB projection
        const vol_spec                  volchunkspec,    // note: NOT of type volume_chunk_spec
        const proj_chunk_spec_maxlen10  proj_chunk_spec
)
{
    // Get the position within the chunk
    const uivec3 vol_idx = uivec3(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    );
    const uivec3 vol_chunk_shape = volchunkspec.shape;
    const fvec3  pos_000_c       = volchunkspec.pos_000;

    const Tv value = _backproject_CB_getval<Tv,Tp>(
        config,
        pos_000_c,
        vol_chunk_shape,
        proj_chunk_spec,
        vol_idx,
        1.f
    );

    // Write the value to the surface memory and leave
    if (value == (Tv)0) { return; }  // Generally not safe to typecast a number to the half type, but OK when 0.
    Tv* volume = (Tv*)volchunkspec.ptr_000;
    unsigned long vol_idx_addr = (
        volchunkspec.strides_bytes.x * (vol_idx.x) +
        volchunkspec.strides_bytes.y * (vol_idx.y) +
        volchunkspec.strides_bytes.z * (vol_idx.z)
    )/sizeof(Tv);
    atomicAdd(
        &volume[vol_idx_addr],
        value
    );
}


template<
    typename Tv,  // type of volume data. e.g. float or double. It is assumed that the summand_multiplier_surface is the same type!
    typename Tp>  // type of projection data. e.g. float or double
__global__ void backproject_CB_voltex(
        const config_CB_backproject     config,          // any miscellaneous config for the CB projection
        const vol_tex_chunk_spec        voltexchunkspec,
        const cudaSurfaceObject_t       volume_surface,  // CUDA surface object used to read and write to the volume
        const cudaSurfaceObject_t       summand_multiplier_surface,  // if configured to do so, then the backprojection will be multiplied by the corresponding value in this multiplier before being added to the volume
        const proj_chunk_spec_maxlen10  proj_chunk_spec
)
{
    // Get the position within the chunk
    const uivec3 vol_idx = uivec3(
        threadIdx.x + blockIdx.x * blockDim.x,
        threadIdx.y + blockIdx.y * blockDim.y,
        threadIdx.z + blockIdx.z * blockDim.z
    );
    const uivec3 vol_chunk_shape = voltexchunkspec.shape;
    const float voxwid = voltexchunkspec.voxwid;
    const fvec3  pos_000_c       = voltexchunkspec.texpos + (
            fvec3(0.5f, 0.5f, 0.5f) + voltexchunkspec.offsets)*voxwid;

    Tv value = _backproject_CB_getval<Tv,Tp>(
        config,
        pos_000_c,
        vol_chunk_shape,
        proj_chunk_spec,
        vol_idx,
        voxwid
    );

    // Write the value to the surface memory and leave
    if (value == (Tv)0) { return; }  // Generally not safe to typecast a number to the half type, but OK when 0.
    // Multiply by the multiplier, if applicable
    if (config.apply_summand_multiplier) {
        value *= read_from_surf3D<Tv>(summand_multiplier_surface, vol_idx + voltexchunkspec.offsets);
    }
    value += read_from_surf3D<Tv>(volume_surface, vol_idx + voltexchunkspec.offsets);
    write_to_surf3D<Tv>(value, volume_surface, vol_idx + voltexchunkspec.offsets);
}
