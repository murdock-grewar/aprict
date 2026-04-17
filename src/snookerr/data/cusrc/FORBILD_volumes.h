struct FORBILD_Box {
    float rho;
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
};
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


struct FORBILD_ClipAxisSpec {
    fvec3 axis;
    float dotprod;
    unsigned char mode;
};
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
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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
        // Implicitly, v1 is not in the region.
        // Find how many multiples of dv must be added to v1 before it gets into the region
        // At this point in the code, that multiple should be in the interval (0,1]
        if (dp_dv_axis == 0) { return fvec2(0.5f, 1.f); }
        float dp_diff = cas.dotprod - dot_prod(v1, cas.axis);
        float interval_start = dp_diff / dp_dv_axis;
        return fvec2(interval_start, 1.f);
    } else {
        // Implicitly, v2 is not in the region.
        // Find how many multiples of dv must be added to v2 before it gets into the region
        // At this point in the code, that multiple should be in the interval [-1,0)
        if (dp_dv_axis == 0) { return fvec2(0.f, 0.5f); }
        float dp_diff = cas.dotprod - dot_prod(v1, cas.axis);
        float interval_end = dp_diff / dp_dv_axis;
        return fvec2(0.f, interval_end);
    }
}


struct FORBILD_Cone {
    float rho;
    float x;
    float y;
    float z;
    float l;
    float r1;
    float r2;
    fvec3 axis;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone cone
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone cone
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Cone_x {
    float rho;
    float x;
    float y;
    float z;
    float l;
    float r1;
    float r2;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_x cone_x
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_x cone_x
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Cone_y {
    float rho;
    float x;
    float y;
    float z;
    float l;
    float r1;
    float r2;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_y cone_y
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_y cone_y
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Cone_z {
    float rho;
    float x;
    float y;
    float z;
    float l;
    float r1;
    float r2;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_z cone_z
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_z cone_z
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Cylinder {
    float rho;
    float x;
    float y;
    float z;
    float r;
    float l;
    fvec3 axis;
};
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


struct FORBILD_Cylinder_x {
    float rho;
    float x;
    float y;
    float z;
    float r;
    float l;
};
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


struct FORBILD_Cylinder_y {
    float rho;
    float x;
    float y;
    float z;
    float r;
    float l;
};
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

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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


struct FORBILD_Cylinder_z {
    float rho;
    float x;
    float y;
    float z;
    float r;
    float l;
};
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

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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


struct FORBILD_Ellipsoid {
    float rho;
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid ellipsoid
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid ellipsoid
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Ellipsoid_free {
    float rho;
    float x;
    float y;
    float z;
    float dx;
    float dy;
    float dz;
    fvec3 a_x;
    fvec3 a_y;
    fvec3 a_z;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid_free ellipsoid_free
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid_free ellipsoid_free
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Ellipt_Cyl {
    float rho;
    float x;
    float y;
    float z;
    float l;
    float dx;
    float dy;
    float dz;
    fvec3 axis;
    fvec3 a_x;
    fvec3 a_y;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl ellipt_cyl
)
{
    // TODO
    return false;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl ellipt_cyl
)
{
    // TODO
    return fvec2(0.f, 0.f);
}


struct FORBILD_Sphere {
    float rho;
    float x;
    float y;
    float z;
    float r;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Sphere sphere
)
{
    fvec3 pos = fvec3(sphere.x, sphere.y, sphere.z);
    float d = norm_fvec3(pos - v);
    return (d < sphere.r);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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


struct FORBILD_Shape {
    unsigned char type;
    FORBILD_ClipAxisSpec cas1;
    FORBILD_ClipAxisSpec cas2;
    FORBILD_ClipAxisSpec cas3;
    FORBILD_Box Box;
    FORBILD_Cone Cone;
    FORBILD_Cone_x Cone_x;
    FORBILD_Cone_y Cone_y;
    FORBILD_Cone_z Cone_z;
    FORBILD_Cylinder Cylinder;
    FORBILD_Cylinder_x Cylinder_x;
    FORBILD_Cylinder_y Cylinder_y;
    FORBILD_Cylinder_z Cylinder_z;
    FORBILD_Ellipsoid Ellipsoid;
    FORBILD_Ellipsoid_free Ellipsoid_free;
    FORBILD_Ellipt_Cyl Ellipt_Cyl;
    FORBILD_Sphere Sphere;
};
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Shape shape
)
{
    if (!_in_region(v, shape.cas1)) { return false; }
    if (!_in_region(v, shape.cas2)) { return false; }
    if (!_in_region(v, shape.cas3)) { return false; }

    switch (shape.type) {
        case 0:
            return false;
        case 1:
            return _in_region(v, shape.Box);
        case 2:
            return _in_region(v, shape.Cone);
        case 3:
            return _in_region(v, shape.Cone_x);
        case 4:
            return _in_region(v, shape.Cone_y);
        case 5:
            return _in_region(v, shape.Cone_z);
        case 6:
            return _in_region(v, shape.Cylinder);
        case 7:
            return _in_region(v, shape.Cylinder_x);
        case 8:
            return _in_region(v, shape.Cylinder_y);
        case 9:
            return _in_region(v, shape.Cylinder_z);
        case 10:
            return _in_region(v, shape.Ellipsoid);
        case 11:
            return _in_region(v, shape.Ellipsoid_free);
        case 12:
            return _in_region(v, shape.Ellipt_Cyl);
        case 13:
            return _in_region(v, shape.Sphere);
        default:
            return false;
    }
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Shape shape
)
{
    fvec2 interval;
    float minf, maxf;
    interval = FORBILD_clip_line_segment(v1, v2, shape.cas1);
    minf = interval.x;  // first  component
    maxf = interval.y;  // second component
    interval = FORBILD_clip_line_segment(v1, v2, shape.cas2);
    minf = max(minf, interval.x);  // first  component
    maxf = min(maxf, interval.y);  // second component
    interval = FORBILD_clip_line_segment(v1, v2, shape.cas3);
    minf = max(minf, interval.x);  // first  component
    maxf = min(maxf, interval.y);  // second component

    switch (shape.type) {
        case 0:
            return fvec2(0.f, 0.f);
        case 1:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Box);
            break;
        case 2:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cone);
            break;
        case 3:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cone_x);
            break;
        case 4:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cone_y);
            break;
        case 5:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cone_z);
            break;
        case 6:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cylinder);
            break;
        case 7:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cylinder_x);
            break;
        case 8:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cylinder_y);
            break;
        case 9:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Cylinder_z);
            break;
        case 10:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Ellipsoid);
            break;
        case 11:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Ellipsoid_free);
            break;
        case 12:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Ellipt_Cyl);
            break;
        case 13:
            interval = FORBILD_clip_line_segment(v1, v2, shape.Sphere);
            break;
        default:
            interval = fvec2(0.f, 0.f);
            break;
    }
    minf = max(minf, interval.x);  // first  component
    maxf = min(maxf, interval.y);  // second component
    minf = min(minf, maxf);  // ensure maxf is greater than or equal to minf
    return fvec2(minf, maxf);
}

template<typename Ts>
__device__ float FORBILD_get_rho(Ts shape)
{
    return shape.rho;
}
template<>
__device__ float FORBILD_get_rho<FORBILD_Shape>(FORBILD_Shape shape)
{
    switch (shape.type) {
        case 0:
            return 0.f;
        case 1:
            return shape.Box.rho;
        case 2:
            return shape.Cone.rho;
        case 3:
            return shape.Cone_x.rho;
        case 4:
            return shape.Cone_y.rho;
        case 5:
            return shape.Cone_z.rho;
        case 6:
            return shape.Cylinder.rho;
        case 7:
            return shape.Cylinder_x.rho;
        case 8:
            return shape.Cylinder_y.rho;
        case 9:
            return shape.Cylinder_z.rho;
        case 10:
            return shape.Ellipsoid.rho;
        case 11:
            return shape.Ellipsoid_free.rho;
        case 12:
            return shape.Ellipt_Cyl.rho;
        case 13:
            return shape.Sphere.rho;
        default:
            return 0.f;
    }
}

// A template method for all the shapes to use, including FORBILD_Shape
template<
    typename Tp,  // type of projection data. e.g. float or double
    typename Ts
>
__device__ Tp exact_line_integral(
        const fvec3                 v1,
        const fvec3                 v2,
        const Ts                    shape
)
{
    const fvec3 dv = v2 - v1;
    const float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) { return (Tp)0; }  // Typically don't want to cast to type half, but it's OK for 0.
    const fvec2 interval = FORBILD_clip_line_segment(v1, v2, shape);
    const Tp integ = FORBILD_get_rho(shape) * ((interval.y - interval.x)*dv_norm);
    return integ;
}

// A template method for all the shapes to use, including FORBILD_Shape
template<typename Ts>
__device__ float exact_line_integral_prioritise_last(
        const fvec3                 v1,
        const fvec3                 v2,
        const Ts*                   shapes,
        const unsigned int          num_shapes
)
{
    // Base case
    if (num_shapes == 0) {
        return 0.f;
    }
    // Return if line segment has length 0
    const fvec3 dv = v2 - v1;
    const float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) {
        return 0.f;
    }

    // To avoid recursion, use a loop to find the first shape (beginning from the end)
    // which has any overlap with this line
    fvec2 interval;
    unsigned int new_num_shapes;
    for (new_num_shapes = num_shapes-1; true; new_num_shapes--) {
        interval = FORBILD_clip_line_segment(v1, v2, shapes[new_num_shapes]);
        if (interval.y > interval.x) {
            break;
        }
        if (new_num_shapes == 0) {
            // No intersection was found with any shape
            return 0.f;
        }
    }
    const Ts shape = shapes[new_num_shapes];
    float val = (interval.y - interval.x) * FORBILD_get_rho(shape) * dv_norm;
    if (new_num_shapes > 0) {
        val += exact_line_integral_prioritise_last(
            v1, v1 + dv*interval.x, shapes, new_num_shapes);
        val += exact_line_integral_prioritise_last(
            v1 + dv*interval.y, v2, shapes, new_num_shapes);
    }
    return val;
}


