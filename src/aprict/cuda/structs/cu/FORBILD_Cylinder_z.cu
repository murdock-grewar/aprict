__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cylinder_z cylinder_z
)
{
    const fvec3 pos = fvec3(cylinder_z.x, cylinder_z.y, cylinder_z.z);
    const fvec3 v_rel = v - pos;
    if (abs(v_rel.z) > cylinder_z.l/2.f) { return false; }
    if (v_rel.x*v_rel.x + v_rel.y*v_rel.y > cylinder_z.r*cylinder_z.r) { return false; }
    return true;
}

__device__ bbox get_bbox(
    FORBILD_Cylinder_z c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.r;
    b.xyz_max.x = c.x + c.r;
    b.xyz_min.y = c.y - c.r;
    b.xyz_max.y = c.y + c.r;
    b.xyz_min.z = c.z - c.l/2.f;
    b.xyz_max.z = c.z + c.l/2.f;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cylinder_z cylinder_z
)
{
    FORBILD_Cylinder cylinder;
    cylinder.x = cylinder_z.x;
    cylinder.y = cylinder_z.y;
    cylinder.z = cylinder_z.z;
    cylinder.r = cylinder_z.r;
    cylinder.l = cylinder_z.l;
    cylinder.axis = fvec3(0.f, 0.f, 1.f);
    return FORBILD_clip_line_segment(v1, v2, cylinder);
}
