__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl ecyl
)
{
    const fvec3 pos = fvec3(ecyl.x, ecyl.y, ecyl.z);
    fvec3 v_rel = v - pos;
    fvec3 az_normalised = ecyl.axis / norm_fvec3(ecyl.axis);
    float vz = dot_prod(v_rel, az_normalised);
    if (abs(vz) > ecyl.l/2.f) {
        return false;
    }

    fvec3 ax_normalised = ecyl.a_x / norm_fvec3(ecyl.a_x);
    float vx = dot_prod(v_rel, ax_normalised);
    fvec3 ay_normalised = ecyl.a_y / norm_fvec3(ecyl.a_y);
    float vy = dot_prod(v_rel, ay_normalised);
    vx /= ecyl.dx;
    vy /= ecyl.dy;
    return (vx*vx + vy*vy <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipt_Cyl ecyl
)
{
    fvec3 ax = ecyl.a_x / norm_fvec3(ecyl.a_x);
    fvec3 ay = ecyl.a_y / norm_fvec3(ecyl.a_y);
    fvec3 az = ecyl.axis / norm_fvec3(ecyl.axis);
    bbox b;
    float d = abs(ecyl.dx*ax.x) + abs(ecyl.dy*ay.x) + abs(ecyl.l/2.f*az.x);
    b.xyz_min.x = ecyl.x - d;
    b.xyz_max.x = ecyl.x + d;
    d = abs(ecyl.dx*ax.y) + abs(ecyl.dy*ay.y) + abs(ecyl.l/2.f*az.y);
    b.xyz_min.y = ecyl.y - d;
    b.xyz_max.y = ecyl.y + d;
    d = abs(ecyl.dx*ax.z) + abs(ecyl.dy*ay.z) + abs(ecyl.l/2.f*az.z);
    b.xyz_min.z = ecyl.z - d;
    b.xyz_max.z = ecyl.z + d;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl ecyl
)
{
    const fvec3 pos = fvec3(ecyl.x, ecyl.y, ecyl.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 ax_normalised = ecyl.a_x / norm_fvec3(ecyl.a_x);
    fvec3 ay_normalised = ecyl.a_y / norm_fvec3(ecyl.a_y);
    fvec3 az_normalised = ecyl.axis / norm_fvec3(ecyl.axis);
    float v1x = dot_prod(v1_rel, ax_normalised);
    float v1y = dot_prod(v1_rel, ay_normalised);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v2x = dot_prod(v2_rel, ax_normalised);
    float v2y = dot_prod(v2_rel, ay_normalised);
    float v2z = dot_prod(v2_rel, az_normalised);
    v1x /= ecyl.dx;
    v1y /= ecyl.dy;
    v2x /= ecyl.dx;
    v2y /= ecyl.dy;
    FORBILD_Cylinder_z cylinder_z;
    cylinder_z.x = 0.f;
    cylinder_z.y = 0.f;
    cylinder_z.z = 0.f;
    cylinder_z.r = 1.f;
    cylinder_z.l = ecyl.l;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), cylinder_z);
}
