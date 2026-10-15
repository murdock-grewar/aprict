__device__ bool _in_region(
    fvec3 v,
    FORBILD_ClipAxisSpec cas
)
{
    unsigned short mode = cas.mode;
    if (mode == 0) { return true; }
    if (mode == 1) {
        return (dot_prod(v, cas.axis) <= cas.dotprod);
    }
    if (mode == 2) {
        return (dot_prod(v, cas.axis) > cas.dotprod);
    }
    return true;  // fallback if the mode is not recognized
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the correct side of the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_ClipAxisSpec cas
)
{
    if (cas.mode == 0) {
        return fvec2(0.f, 1.f);
    }
    bool v1_in_region = _in_region(v1, cas);
    bool v2_in_region = _in_region(v2, cas);
    // The clipped region is convex, since it's half of R^3, split by the clipping plane.
    // So, if both points are in the region, then the whole interval is.
    // If neither point is in the region, then none of the interval is.
    if (v1_in_region && v2_in_region) {
        return fvec2(0.f, 1.f);  // Totally in range
    }
    if ((!v1_in_region) && (!v2_in_region)) {
        return fvec2(0.f, 0.f);  // Totally out of range
    }
    fvec3 dv = v2 - v1;

    // Edge case: dot_prod(dv, cas.axis) == 0
    // In this case, it should be that both v1 and v2 are in the region, so we should have handled that...
    // But if there's some floating point edge case where this occurs, then let's choose
    // either [0, 0.5] or [0.5, 1] according to which point is in the region.
    float dp_dv_axis = dot_prod(dv, cas.axis);

    if (v2_in_region) {
        // Implicitly, v1 is NOT in the region and v2 is in the region.
        // Find how many multiples of dv must be added to v1 before it gets into the region
        // At this point in the code, that multiple should be in the interval (0,1]
        if (dp_dv_axis == 0) { return fvec2(0.5f, 1.f); }
        float dp_diff = cas.dotprod - dot_prod(v1, cas.axis);
        float interval_start = dp_diff / dp_dv_axis;
        return fvec2(max(0.f, min(1.f, interval_start)), 1.f);
    } else {
        // Implicitly, v1 is in the region and v2 is NOT in the region.
        // Find how many multiples of dv must be added to v1 before it leaves the region
        // At this point in the code, that multiple should be in the interval (0,1]
        if (dp_dv_axis == 0) { return fvec2(0.f, 0.5f); }
        float dp_diff = cas.dotprod - dot_prod(v1, cas.axis);
        float interval_end = dp_diff / dp_dv_axis;
        return fvec2(0.f, max(0.f, min(1.f, interval_end)));
    }
}
