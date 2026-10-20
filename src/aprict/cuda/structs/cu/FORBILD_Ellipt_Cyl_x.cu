__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl_x ecylx
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecylx.rho;
    ecyl.x = ecylx.x;
    ecyl.y = ecylx.y;
    ecyl.z = ecylx.z;
    ecyl.l = ecylx.l;
    ecyl.dx = ecylx.dy;
    ecyl.dy = ecylx.dz;
    ecyl.axis = fvec3(1.f,0.f,0.f);
    ecyl.a_x = fvec3(0.f,1.f,0.f);
    ecyl.a_y = fvec3(0.f,0.f,1.f);
    return _in_region(v, ecyl);
}

__device__ bbox get_bbox(
    FORBILD_Ellipt_Cyl_x ecx
)
{
    bbox b;
    b.xyz_min.x = ecx.x - ecx.l/2.f;
    b.xyz_max.x = ecx.x + ecx.l/2.f;
    b.xyz_min.y = ecx.y - ecx.dy;
    b.xyz_max.y = ecx.y + ecx.dy;
    b.xyz_min.z = ecx.z - ecx.dz;
    b.xyz_max.z = ecx.z + ecx.dz;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl_x ecylx
)
{
    FORBILD_Ellipt_Cyl ecyl;
    ecyl.rho = ecylx.rho;
    ecyl.x = ecylx.x;
    ecyl.y = ecylx.y;
    ecyl.z = ecylx.z;
    ecyl.l = ecylx.l;
    ecyl.dx = ecylx.dy;
    ecyl.dy = ecylx.dz;
    ecyl.axis = fvec3(1.f,0.f,0.f);
    ecyl.a_x = fvec3(0.f,1.f,0.f);
    ecyl.a_y = fvec3(0.f,0.f,1.f);
    return FORBILD_clip_line_segment(v1, v2, ecyl);
}
