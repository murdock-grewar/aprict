__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cylinder_y cylinder_y
)
{
    const fvec3 pos = fvec3(cylinder_y.x, cylinder_y.y, cylinder_y.z);
    const fvec3 v_rel = v - pos;
    if (abs(v_rel.y) > cylinder_y.l/2.f) { return false; }
    if (v_rel.x*v_rel.x + v_rel.z*v_rel.z > cylinder_y.r*cylinder_y.r) { return false; }
    return true;
}

__device__ bbox get_bbox(
    FORBILD_Cylinder_y c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.r;
    b.xyz_max.x = c.x + c.r;
    b.xyz_min.y = c.y - c.l/2.f;
    b.xyz_max.y = c.y + c.l/2.f;
    b.xyz_min.z = c.z - c.r;
    b.xyz_max.z = c.z + c.r;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cylinder_y cylinder_y
)
{
    FORBILD_Cylinder cylinder;
    cylinder.x = cylinder_y.x;
    cylinder.y = cylinder_y.y;
    cylinder.z = cylinder_y.z;
    cylinder.r = cylinder_y.r;
    cylinder.l = cylinder_y.l;
    cylinder.axis = fvec3(0.f, 1.f, 0.f);
    return FORBILD_clip_line_segment(v1, v2, cylinder);
}
