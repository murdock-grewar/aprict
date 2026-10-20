__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl_z ecylz
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecylz.rho;
    ecyl.x = ecylz.x;
    ecyl.y = ecylz.y;
    ecyl.z = ecylz.z;
    ecyl.l = ecylz.l;
    ecyl.dx = ecylz.dx;
    ecyl.dy = ecylz.dy;
    ecyl.axis = fvec3(0.f,0.f,1.f);
    ecyl.a_x = fvec3(1.f,0.f,0.f);
    ecyl.a_y = fvec3(0.f,1.f,0.f);
    return _in_region(v, ecyl);
}

__device__ bbox get_bbox(
    FORBILD_Ellipt_Cyl_z ecz
)
{
    bbox b;
    b.xyz_min.x = ecz.x - ecz.dx;
    b.xyz_max.x = ecz.x + ecz.dx;
    b.xyz_min.y = ecz.y - ecz.dy;
    b.xyz_max.y = ecz.y + ecz.dy;
    b.xyz_min.z = ecz.z - ecz.l/2.f;
    b.xyz_max.z = ecz.z + ecz.l/2.f;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl_z ecylz
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecylz.rho;
    ecyl.x = ecylz.x;
    ecyl.y = ecylz.y;
    ecyl.z = ecylz.z;
    ecyl.l = ecylz.l;
    ecyl.dx = ecylz.dx;
    ecyl.dy = ecylz.dy;
    ecyl.axis = fvec3(0.f,0.f,1.f);
    ecyl.a_x = fvec3(1.f,0.f,0.f);
    ecyl.a_y = fvec3(0.f,1.f,0.f);
    return FORBILD_clip_line_segment(v1, v2, ecyl);
}
