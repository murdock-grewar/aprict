__device__ bool _in_region(
    fvec3 v,
    FORBILD_Box box
)
{
    if (abs(v.x - box.x) > box.dx/2.f) { return false; }
    if (abs(v.y - box.y) > box.dy/2.f) { return false; }
    if (abs(v.z - box.z) > box.dz/2.f) { return false; }
    return true;
}

__device__ bbox get_bbox(
    FORBILD_Box box
)
{
    fvec3 v1;
    v1.x = box.x - box.dx/2.f;
    v1.y = box.y - box.dy/2.f;
    v1.z = box.z - box.dz/2.f;
    fvec3 v2;
    v2.x = box.x + box.dx/2.f;
    v2.y = box.y + box.dy/2.f;
    v2.z = box.z + box.dz/2.f;
    return bbox(v1, v2);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the Box.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Box box
)
{
    bool v1_in_region = _in_region(v1, box);
    bool v2_in_region = _in_region(v2, box);
    if (v1_in_region && v2_in_region) {
        return fvec2(0.f, 1.f);
    }

    // Create the full interval, then shrink according to all three axes
    fvec2 interval = fvec2(0.f, 1.f);
    fvec3 dv = v2 - v1;
    float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) { return fvec2(0.f, 0.f); }
    if (dv.x == 0) {
        if (abs(v1.x - box.x) > box.dx/2.f) {
            return fvec2(0.f, 0.f);
        }
    } else {
        float lam1 = (box.x - box.dx/2.f - v1.x)/dv.x;
        float lam2 = (box.x + box.dx/2.f - v1.x)/dv.x;
        interval.x = max(interval.x, min(lam1, lam2));
        interval.y = min(interval.y, max(lam1, lam2));
    }
    if (dv.y == 0) {
        if (abs(v1.y - box.y) > box.dy/2.f) {
            return fvec2(0.f, 0.f);
        }
    } else {
        float lam1 = (box.y - box.dy/2.f - v1.y)/dv.y;
        float lam2 = (box.y + box.dy/2.f - v1.y)/dv.y;
        interval.x = max(interval.x, min(lam1, lam2));
        interval.y = min(interval.y, max(lam1, lam2));
    }
    if (dv.z == 0) {
        if (abs(v1.z - box.z) > box.dz/2.f) {
            return fvec2(0.f, 0.f);
        }
    } else {
        float lam1 = (box.z - box.dz/2.f - v1.z)/dv.z;
        float lam2 = (box.z + box.dz/2.f - v1.z)/dv.z;
        interval.x = max(interval.x, min(lam1, lam2));
        interval.y = min(interval.y, max(lam1, lam2));
    }
    interval.x = max(0.f, min(1.f, interval.x));
    interval.y = max(interval.x, interval.y);
    return interval;
}
