__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid ellipsoid
)
{
    const fvec3 pos = fvec3(ellipsoid.x, ellipsoid.y, ellipsoid.z);
    fvec3 v_rel = v - pos;
    v_rel.x /= ellipsoid.dx;
    v_rel.y /= ellipsoid.dy;
    v_rel.z /= ellipsoid.dz;
    return (norm_fvec3(v_rel) <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipsoid e
)
{
    bbox b;
    b.xyz_min.x = e.x - e.dx;
    b.xyz_max.x = e.x + e.dx;
    b.xyz_min.y = e.y - e.dy;
    b.xyz_max.y = e.y + e.dy;
    b.xyz_min.z = e.z - e.dz;
    b.xyz_max.z = e.z + e.dz;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid ellipsoid
)
{
    const fvec3 pos = fvec3(ellipsoid.x, ellipsoid.y, ellipsoid.z);
    fvec3 v1_rel = v1 - pos;
    v1_rel.x /= ellipsoid.dx;
    v1_rel.y /= ellipsoid.dy;
    v1_rel.z /= ellipsoid.dz;
    fvec3 v2_rel = v2 - pos;
    v2_rel.x /= ellipsoid.dx;
    v2_rel.y /= ellipsoid.dy;
    v2_rel.z /= ellipsoid.dz;
    FORBILD_Sphere sphere;
    sphere.x = 0.f;
    sphere.y = 0.f;
    sphere.z = 0.f;
    sphere.r = 1.f;
    return FORBILD_clip_line_segment(v1_rel, v2_rel, sphere);
}
