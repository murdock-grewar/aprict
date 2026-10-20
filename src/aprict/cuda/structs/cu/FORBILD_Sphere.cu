__device__ bool _in_region(
    fvec3 v,
    FORBILD_Sphere sphere
)
{
    fvec3 pos = fvec3(sphere.x, sphere.y, sphere.z);
    float d = norm_fvec3(pos - v);
    return (d < sphere.r);
}

__device__ bbox get_bbox(
    FORBILD_Sphere sphere
)
{
    fvec3 v1;
    v1.x = sphere.x - sphere.r;
    v1.y = sphere.y - sphere.r;
    v1.z = sphere.z - sphere.r;
    fvec3 v2;
    v2.x = sphere.x + sphere.r;
    v2.y = sphere.y + sphere.r;
    v2.z = sphere.z + sphere.r;
    return bbox(v1, v2);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Sphere sphere
)
{
    // The sphere is a convex object
    if ((v1.x - sphere.x > sphere.r) &&
        (v2.x - sphere.x > sphere.r)
    ) { return fvec2(0.f, 0.f); }
    if ((v1.y - sphere.y > sphere.r) &&
        (v2.y - sphere.y > sphere.r)
    ) { return fvec2(0.f, 0.f); }
    if ((v1.z - sphere.z > sphere.r) &&
        (v2.z - sphere.z > sphere.r)
    ) { return fvec2(0.f, 0.f); }
    if ((v1.x - sphere.x < -sphere.r) &&
        (v2.x - sphere.x < -sphere.r)
    ) { return fvec2(0.f, 0.f); }
    if ((v1.y - sphere.y < -sphere.r) &&
        (v2.y - sphere.y < -sphere.r)
    ) { return fvec2(0.f, 0.f); }
    if ((v1.z - sphere.z < -sphere.r) &&
        (v2.z - sphere.z < -sphere.r)
    ) { return fvec2(0.f, 0.f); }

    bool v1_in_region = _in_region(v1, sphere);
    bool v2_in_region = _in_region(v2, sphere);
    if (v1_in_region && v2_in_region) {
        return fvec2(0.f, 1.f);  // Totally in range
    }

    fvec3 dv = v2 - v1;
    float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) { return fvec2(0.f, 0.f); }

    // Calculate the point along the infinite line which is closest to the center of the sphere
    // Solving for d in:
    // (pos - d) . dv = 0, with d = v1 + m*dv
    // ==> (v1 + m*dv - pos) . dv = 0
    // ==> (v1 - pos) . dv + m (dv.dv) = 0
    // ==> m (dv.dv) = (pos - v1) . dv
    // ==> m = ((pos - v1) . dv) / (dv.dv)
    fvec3 pos = fvec3(sphere.x, sphere.y, sphere.z);
    float m = dot_prod((pos - v1)/dv_norm, dv/dv_norm);
    // Now 'm' is the point (possibly in the interval [0,1]) of closest approach
    fvec3 d = v1 + (dv*m);
    float dist = norm_fvec3(pos - d);
    if (dist > sphere.r) { return fvec2(0.f, 0.f); }
    float interval_pm = sqrt(sphere.r*sphere.r - dist*dist)/dv_norm;
    return fvec2(
        max(0.f, min(1.f, m - interval_pm)),
        max(0.f, min(1.f, m + interval_pm))
    );
}
