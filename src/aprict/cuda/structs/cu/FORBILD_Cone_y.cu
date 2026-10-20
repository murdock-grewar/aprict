__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_y cone_y
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_y.x, cone_y.y, cone_y.z);
    fvec3 v_rel = v - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_y.l;
    cz.r1 = cone_y.r1;
    cz.r2 = cone_y.r2;
    return _in_region(fvec3(v_rel.x, v_rel.z, v_rel.y), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone_y cone_y
)
{
    FORBILD_Cylinder_y c;
    c.x = cone_y.x;
    c.y = cone_y.y;
    c.z = cone_y.z;
    c.r = max(cone_y.r1, cone_y.r2);
    c.l = cone_y.l;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_y cone_y
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_y.x, cone_y.y, cone_y.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_y.l;
    cz.r1 = cone_y.r1;
    cz.r2 = cone_y.r2;
    return FORBILD_clip_line_segment(fvec3(v1_rel.x, v1_rel.z, v1_rel.y), fvec3(v2_rel.x, v2_rel.z, v2_rel.y), cz);
}
