__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cylinder cylinder
)
{
    // Convert to cyl_rho and cyl_z coords
    const fvec3 pos = fvec3(cylinder.x, cylinder.y, cylinder.z);
    float cyl_z = dot_prod(v - pos, cylinder.axis/norm_fvec3(cylinder.axis));
    if (abs(cyl_z) > cylinder.l/2.f) { return false; }
    float nv = norm_fvec3(v - pos);
    float cyl_rho_sq = max(0.f, nv*nv - cyl_z*cyl_z);
    return (cyl_rho_sq <= cylinder.r*cylinder.r);
}

__device__ bbox get_bbox(
    FORBILD_Cylinder cylinder
)
{
    // We just need to construct any rectangular prism fitting the cylinder,
    // then take the minimum/maximum x,y,z coords of all 8 points.
    const fvec3 pos = fvec3(cylinder.x, cylinder.y, cylinder.z);
    fvec3 axis_normed = cylinder.axis/norm_fvec3(cylinder.axis);
    float hl = cylinder.l/2.f;

    // Get the midpoints of the circular caps
    fvec3 end1 = pos + axis_normed*hl;
    fvec3 end2 = pos - axis_normed*hl;

    // Get azimuthal angle of cylinder axis.
    float axis_aziang = atan2f(cylinder.axis.y, cylinder.axis.x);
    // Get polar angle
    float cos_pol = axis_normed.z;
    float sin_pol = sqrt(max(0.f, 1 - cos_pol*cos_pol));

    // Instantiate bounding box
    bbox b;

    // Compute min/max coords
    float dx = abs(sin(axis_aziang) * cylinder.r) + abs(cos(axis_aziang) * cylinder.r * cos_pol);
    float dy = abs(cos(axis_aziang) * cylinder.r) + abs(sin(axis_aziang) * cylinder.r * cos_pol);
    float dz = abs(sin_pol * cylinder.r);
    b.xyz_min.x = min(end1.x, end2.x) - dx;
    b.xyz_min.y = min(end1.y, end2.y) - dy;
    b.xyz_min.z = min(end1.z, end2.z) - dz;
    b.xyz_max.x = max(end1.x, end2.x) + dx;
    b.xyz_max.y = max(end1.y, end2.y) + dy;
    b.xyz_max.z = max(end1.z, end2.z) + dz;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cylinder cylinder
)
{
    bool v1_in_region = _in_region(v1, cylinder);
    bool v2_in_region = _in_region(v2, cylinder);
    if (v1_in_region && v2_in_region) {
        return fvec2(0.f, 1.f);  // Totally in range
    }

    fvec3 dv = v2 - v1;
    float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) { return fvec2(0.f, 0.f); }

    // Interval to be in bounds of the cylinder z
    const fvec3 pos = fvec3(cylinder.x, cylinder.y, cylinder.z);
    const fvec3 axis_normalised = cylinder.axis/norm_fvec3(cylinder.axis);
    const float v1_cyl_z = dot_prod(v1 - pos, axis_normalised);
    // const float nv1 = norm_fvec3(v1 - pos);
    // const float v1_cyl_rho_sq = max(0.f, nv1*nv1 - v1_cyl_z*v1_cyl_z);
    const float v2_cyl_z = dot_prod(v2 - pos, axis_normalised);
    const float dv_cyl_z = v2_cyl_z - v1_cyl_z;
    fvec2 interval;
    if (dv_cyl_z == 0) {
        if (abs(v1_cyl_z) > cylinder.l/2.f) {
            return fvec2(0.f, 0.f);
        }
        interval.x = 0.f;
        interval.y = 1.f;
    } else {
        float lam1 = ( cylinder.l/2.f - v1_cyl_z)/dv_cyl_z;
        float lam2 = (-cylinder.l/2.f - v1_cyl_z)/dv_cyl_z;
        interval.x = max(0.f, min(lam1, lam2));
        interval.y = min(1.f, max(lam1, lam2));
        interval.y = max(interval.x, interval.y);
        if (interval.y == interval.x) {
            return fvec2(0.f, 0.f);
        }
    }

    // Interval to be in bounds of the cylinder radius
    // Will project v1, v2 into the plane orthogonal to cylinder z axis and containing cylinder pos
    fvec3 v1_projected = v1 - axis_normalised*dot_prod(v1 - pos, axis_normalised);
    fvec3 v2_projected = v2 - axis_normalised*dot_prod(v2 - pos, axis_normalised);
    fvec3 dv_projected = dv - axis_normalised*dot_prod(dv, axis_normalised);
    float dv_projected_norm = norm_fvec3(dv_projected);
    // Calculate the point along the infinite line which is closest to the center of the sphere
    // Solving for d in:
    // (pos - d) . dv = 0, with d = v1 + m*dv
    // ==> (v1 + m*dv - pos) . dv = 0
    // ==> (v1 - pos) . dv + m (dv.dv) = 0
    // ==> m (dv.dv) = (pos - v1) . dv
    // ==> m = ((pos - v1) . dv) / (dv.dv)
    float m = dot_prod((pos - v1_projected)/dv_projected_norm, dv_projected/dv_projected_norm);
    // Now 'm' is the point (possibly in the interval [0,1]) of closest approach
    fvec3 d = v1_projected + (dv_projected*m);
    float dist = norm_fvec3(pos - d);
    if (dist > cylinder.r) { return fvec2(0.f, 0.f); }
    float interval_pm = sqrt(cylinder.r*cylinder.r - dist*dist)/dv_projected_norm;
    fvec2 interval_radius = fvec2(
        max(0.f, min(1.f, m - interval_pm)),
        max(0.f, min(1.f, m + interval_pm))
    );
    interval.x = max(interval.x, interval_radius.x);
    interval.y = min(interval.y, interval_radius.y);
    interval.x = max(0.f, interval.x);
    interval.y = max(interval.x, interval.y);
    return interval;
}
