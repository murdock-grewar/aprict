struct det_spec {
    fvec3 pos;
    fvec3 u;
    fvec3 v;
    unsigned short nu;
    unsigned short nv;
    unsigned char type;
};

struct proj_spec {
    fvec3 sourcepos;
    det_spec dspec;
};
/*
 * In order to mantain O(N^3) scaling in the backprojection, we need a finite, closed-form equation
 * which maps a voxel to its projection onto the detector, in terms of the detector indices.
 *
 * (The alterantive is to run a thread for each detector index, but the lack of atomic addition
 * for surfaces/textures leads to the need to use a lot of global memory atomicAdds for large
 * volumes, which is unacceptably slow. Maybe it's possible to make the memory accesses more
 * efficient for that case; I'm not sure.)
 */

__device__ fvec2 get_projected_detector_coords(
        const fvec3       point,
        const proj_spec   projspec
)
{
    if (projspec.dspec.type == 0) {
        /* Solve equation for p, lambda:
         * p . n = detector . n
         * p = sourcepos + lambda(point - sourcepos)
         * ==> detector . n = sourcepos . n + lambda (point - sourcepos) . n
         * ==> (detector - sourcepos) . n = lambda (point - sourcepos) . n
         * ==> lambda = [(detector - sourcepos) . n] / [(point - sourcepos) . n]
         * ==> p = sourcepos + lambda(point - sourcepos) with lambda as above
         */
        // Compute actual point of intersection, in 3D space
        const fvec3 n = cross_prod(projspec.dspec.u, projspec.dspec.v);
        fvec3 p = projspec.sourcepos + (point - projspec.sourcepos) * (
            dot_prod(projspec.dspec.pos - projspec.sourcepos, n) /
            dot_prod(point              - projspec.sourcepos, n)
        );
        // Compute the point's detector coordinates, from the center.
        p = p - projspec.dspec.pos;
        const float unorm = norm_fvec3(projspec.dspec.u);
        const float vnorm = norm_fvec3(projspec.dspec.v);
        return fvec2(
            dot_prod(projspec.dspec.u, p)/unorm/unorm,
            dot_prod(projspec.dspec.v, p)/vnorm/vnorm
        );
    } else {
        // UNIMPLEMENTED!!
        return fvec2(0.f, 0.f);
    }
}

struct proj_chunk_spec_maxlen10 {
    proj_spec projspecs[10];
    unsigned char numprojs;
    unsigned long projmemaddr;
    ulvec3 projmemstrides;
};

struct proj_tex_chunk_spec_maxlen10 {
    proj_spec projspecs[10];
    unsigned char numprojs;
    unsigned int tex_start_layer;
};

struct vol_spec {
    unsigned long ptr_000;
    ulvec3 strides_bytes;
    uivec3 shape;
    fvec3 pos_000;
};

struct vol_tex_chunk_spec {
    uivec3 offsets;
    uivec3 shape;
    fvec3 texpos;
};

struct volume_chunk_spec {
    unsigned short idx_xstart;
    unsigned short idx_xstop;
    unsigned short idx_ystart;
    unsigned short idx_ystop;
    unsigned short idx_zstart;
    unsigned short idx_zstop;
};

