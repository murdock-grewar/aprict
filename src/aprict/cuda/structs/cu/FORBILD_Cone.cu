__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone cone
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone.x, cone.y, cone.z);
    fvec3 v_rel = v - pos;
    fvec3 az_normalised = cone.axis / norm_fvec3(cone.axis);
    float vz = dot_prod(v_rel, az_normalised);
    float vx = sqrt(max(0.f, (v_rel.x*v_rel.x + v_rel.y*v_rel.y + v_rel.z*v_rel.z) - vz*vz));
    float vy = 0.f;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone.l;
    cz.r1 = cone.r1;
    cz.r2 = cone.r2;
    return _in_region(fvec3(vx, vy, vz), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone cone
)
{
    FORBILD_Cylinder c;
    c.x = cone.x;
    c.y = cone.y;
    c.z = cone.z;
    c.r = max(cone.r1, cone.r2);
    c.l = cone.l;
    c.axis = cone.axis;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone cone
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone.x, cone.y, cone.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 az_normalised = cone.axis / norm_fvec3(cone.axis);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v1x = sqrt(max(0.f, (v1_rel.x*v1_rel.x + v1_rel.y*v1_rel.y + v1_rel.z*v1_rel.z) - v1z*v1z));
    float v1y = 0.f;
    float v2z = dot_prod(v2_rel, az_normalised);
    float v2x, v2y;
    if (v1x == 0) {
        v2x = sqrt(max(0.f, (v2_rel.x*v2_rel.x + v2_rel.y*v2_rel.y + v2_rel.z*v2_rel.z) - v2z*v2z));
        v2y = 0.f;
    } else {
        fvec3 xhat = v1_rel - az_normalised*v1z;
        xhat = xhat / norm_fvec3(xhat);
        v2x = dot_prod(v2_rel, xhat);
        v2y = sqrt(max(0.f, (v2_rel.x*v2_rel.x + v2_rel.y*v2_rel.y + v2_rel.z*v2_rel.z) - v2x*v2x - v2z*v2z));
    }
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone.l;
    cz.r1 = cone.r1;
    cz.r2 = cone.r2;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), cz);
}
