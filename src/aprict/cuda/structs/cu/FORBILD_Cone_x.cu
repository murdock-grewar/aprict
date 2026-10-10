__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_x cone_x
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_x.x, cone_x.y, cone_x.z);
    fvec3 v_rel = v - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_x.l;
    cz.r1 = cone_x.r1;
    cz.r2 = cone_x.r2;
    return _in_region(fvec3(v_rel.z, v_rel.y, v_rel.x), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone_x cone_x
)
{
    FORBILD_Cylinder_x c;
    c.x = cone_x.x;
    c.y = cone_x.y;
    c.z = cone_x.z;
    c.r = max(cone_x.r1, cone_x.r2);
    c.l = cone_x.l;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_x cone_x
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_x.x, cone_x.y, cone_x.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_x.l;
    cz.r1 = cone_x.r1;
    cz.r2 = cone_x.r2;
    return FORBILD_clip_line_segment(fvec3(v1_rel.z, v1_rel.y, v1_rel.x), fvec3(v2_rel.z, v2_rel.y, v2_rel.x), cz);
}
