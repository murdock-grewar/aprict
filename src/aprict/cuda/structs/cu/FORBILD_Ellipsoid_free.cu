__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid_free ef
)
{
    const fvec3 pos = fvec3(ef.x, ef.y, ef.z);
    fvec3 v_rel = v - pos;
    fvec3 ax_normalised = ef.a_x / norm_fvec3(ef.a_x);
    float vx = dot_prod(v_rel, ax_normalised);
    fvec3 ay_normalised = ef.a_y / norm_fvec3(ef.a_y);
    float vy = dot_prod(v_rel, ay_normalised);
    fvec3 az_normalised = ef.a_z / norm_fvec3(ef.a_z);
    float vz = dot_prod(v_rel, az_normalised);
    vx /= ef.dx;
    vy /= ef.dy;
    vz /= ef.dz;
    return (vx*vx + vy*vy + vz*vz <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipsoid_free ef
)
{
    fvec3 ax = ef.a_x / norm_fvec3(ef.a_x);
    fvec3 ay = ef.a_y / norm_fvec3(ef.a_y);
    fvec3 az = ef.a_z / norm_fvec3(ef.a_z);
    bbox b;
    float d = abs(ef.dx*ax.x) + abs(ef.dy*ay.x) + abs(ef.dz*az.x);
    b.xyz_min.x = ef.x - d;
    b.xyz_max.x = ef.x + d;
    d = abs(ef.dx*ax.y) + abs(ef.dy*ay.y) + abs(ef.dz*az.y);
    b.xyz_min.y = ef.y - d;
    b.xyz_max.y = ef.y + d;
    d = abs(ef.dx*ax.z) + abs(ef.dy*ay.z) + abs(ef.dz*az.z);
    b.xyz_min.z = ef.z - d;
    b.xyz_max.z = ef.z + d;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid_free ef
)
{
    const fvec3 pos = fvec3(ef.x, ef.y, ef.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 ax_normalised = ef.a_x / norm_fvec3(ef.a_x);
    fvec3 ay_normalised = ef.a_y / norm_fvec3(ef.a_y);
    fvec3 az_normalised = ef.a_z / norm_fvec3(ef.a_z);
    float v1x = dot_prod(v1_rel, ax_normalised);
    float v1y = dot_prod(v1_rel, ay_normalised);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v2x = dot_prod(v2_rel, ax_normalised);
    float v2y = dot_prod(v2_rel, ay_normalised);
    float v2z = dot_prod(v2_rel, az_normalised);
    v1x /= ef.dx;
    v1y /= ef.dy;
    v1z /= ef.dz;
    v2x /= ef.dx;
    v2y /= ef.dy;
    v2z /= ef.dz;
    FORBILD_Sphere sphere;
    sphere.x = 0.f;
    sphere.y = 0.f;
    sphere.z = 0.f;
    sphere.r = 1.f;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), sphere);
}
