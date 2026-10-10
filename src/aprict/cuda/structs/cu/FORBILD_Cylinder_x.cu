__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cylinder_x cylinder_x
)
{
    const fvec3 pos = fvec3(cylinder_x.x, cylinder_x.y, cylinder_x.z);
    const fvec3 v_rel = v - pos;
    if (abs(v_rel.x) > cylinder_x.l/2.f) { return false; }
    if (v_rel.y*v_rel.y + v_rel.z*v_rel.z > cylinder_x.r*cylinder_x.r) { return false; }
    return true;
}

__device__ bbox get_bbox(
    FORBILD_Cylinder_x c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.l/2.f;
    b.xyz_max.x = c.x + c.l/2.f;
    b.xyz_min.y = c.y - c.r;
    b.xyz_max.y = c.y + c.r;
    b.xyz_min.z = c.z - c.r;
    b.xyz_max.z = c.z + c.r;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cylinder_x cylinder_x
)
{
    FORBILD_Cylinder cylinder;
    cylinder.x = cylinder_x.x;
    cylinder.y = cylinder_x.y;
    cylinder.z = cylinder_x.z;
    cylinder.r = cylinder_x.r;
    cylinder.l = cylinder_x.l;
    cylinder.axis = fvec3(1.f, 0.f, 0.f);
    return FORBILD_clip_line_segment(v1, v2, cylinder);
}
