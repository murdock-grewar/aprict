__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl_y ecyly
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecyly.rho;
    ecyl.x = ecyly.x;
    ecyl.y = ecyly.y;
    ecyl.z = ecyly.z;
    ecyl.l = ecyly.l;
    ecyl.dx = ecyly.dx;
    ecyl.dy = ecyly.dz;
    ecyl.axis = fvec3(0.f,1.f,0.f);
    ecyl.a_x = fvec3(1.f,0.f,0.f);
    ecyl.a_y = fvec3(0.f,0.f,1.f);
    return _in_region(v, ecyl);
}

__device__ bbox get_bbox(
    FORBILD_Ellipt_Cyl_y ecy
)
{
    bbox b;
    b.xyz_min.x = ecy.x - ecy.dx;
    b.xyz_max.x = ecy.x + ecy.dx;
    b.xyz_min.y = ecy.y - ecy.l/2.f;
    b.xyz_max.y = ecy.y + ecy.l/2.f;
    b.xyz_min.z = ecy.z - ecy.dz;
    b.xyz_max.z = ecy.z + ecy.dz;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl_y ecyly
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecyly.rho;
    ecyl.x = ecyly.x;
    ecyl.y = ecyly.y;
    ecyl.z = ecyly.z;
    ecyl.l = ecyly.l;
    ecyl.dx = ecyly.dx;
    ecyl.dy = ecyly.dz;
    ecyl.axis = fvec3(0.f,1.f,0.f);
    ecyl.a_x = fvec3(1.f,0.f,0.f);
    ecyl.a_y = fvec3(0.f,0.f,1.f);
    return FORBILD_clip_line_segment(v1, v2, ecyl);
}
