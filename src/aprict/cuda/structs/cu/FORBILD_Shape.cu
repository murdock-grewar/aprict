__device__ FORBILD_Box _to_Box(const FORBILD_Shape s)
{
    FORBILD_Box box;
    box.rho = s.rho;
    box.x = s.x;
    box.y = s.y;
    box.z = s.z;
    box.dx = s.dx;
    box.dy = s.dy;
    box.dz = s.dz;
    return box;
}

__device__ FORBILD_Cone _to_Cone(const FORBILD_Shape s)
{
    FORBILD_Cone cone;
    cone.rho = s.rho;
    cone.x = s.x;
    cone.y = s.y;
    cone.z = s.z;
    cone.l = s.l;
    cone.r1 = s.r1;
    cone.r2 = s.r2;
    cone.axis = s.axis;
    return cone;
}

__device__ FORBILD_Cone_x _to_Cone_x(const FORBILD_Shape s)
{
    FORBILD_Cone_x cone;
    cone.rho = s.rho;
    cone.x = s.x;
    cone.y = s.y;
    cone.z = s.z;
    cone.l = s.l;
    cone.r1 = s.r1;
    cone.r2 = s.r2;
    return cone;
}

__device__ FORBILD_Cone_y _to_Cone_y(const FORBILD_Shape s)
{
    FORBILD_Cone_y cone;
    cone.rho = s.rho;
    cone.x = s.x;
    cone.y = s.y;
    cone.z = s.z;
    cone.l = s.l;
    cone.r1 = s.r1;
    cone.r2 = s.r2;
    return cone;
}

__device__ FORBILD_Cone_z _to_Cone_z(const FORBILD_Shape s)
{
    FORBILD_Cone_z cone;
    cone.rho = s.rho;
    cone.x = s.x;
    cone.y = s.y;
    cone.z = s.z;
    cone.l = s.l;
    cone.r1 = s.r1;
    cone.r2 = s.r2;
    return cone;
}


__device__ FORBILD_Cylinder _to_Cylinder(const FORBILD_Shape s)
{
    FORBILD_Cylinder c;
    c.rho = s.rho;
    c.x = s.x;
    c.y = s.y;
    c.z = s.z;
    c.r = s.r;
    c.l = s.l;
    c.axis = s.axis;
    return c;
}

__device__ FORBILD_Cylinder_x _to_Cylinder_x(const FORBILD_Shape s)
{
    FORBILD_Cylinder_x c;
    c.rho = s.rho;
    c.x = s.x;
    c.y = s.y;
    c.z = s.z;
    c.r = s.r;
    c.l = s.l;
    return c;
}

__device__ FORBILD_Cylinder_y _to_Cylinder_y(const FORBILD_Shape s)
{
    FORBILD_Cylinder_y c;
    c.rho = s.rho;
    c.x = s.x;
    c.y = s.y;
    c.z = s.z;
    c.r = s.r;
    c.l = s.l;
    return c;
}

__device__ FORBILD_Cylinder_z _to_Cylinder_z(const FORBILD_Shape s)
{
    FORBILD_Cylinder_z c;
    c.rho = s.rho;
    c.x = s.x;
    c.y = s.y;
    c.z = s.z;
    c.r = s.r;
    c.l = s.l;
    return c;
}

__device__ FORBILD_Ellipsoid _to_Ellipsoid(const FORBILD_Shape s)
{
    FORBILD_Ellipsoid e;
    e.rho = s.rho;
    e.x = s.x;
    e.y = s.y;
    e.z = s.z;
    e.dx = s.dx;
    e.dy = s.dy;
    e.dz = s.dz;
    return e;
}

__device__ FORBILD_Ellipsoid_free _to_Ellipsoid_free(const FORBILD_Shape s)
{
    FORBILD_Ellipsoid_free e;
    e.rho = s.rho;
    e.x = s.x;
    e.y = s.y;
    e.z = s.z;
    e.dx = s.dx;
    e.dy = s.dy;
    e.dz = s.dz;
    e.a_x = s.a_x;
    e.a_y = s.a_y;
    e.a_z = s.a_z;
    return e;
}

__device__ FORBILD_Ellipt_Cyl _to_Ellipt_Cyl(const FORBILD_Shape s)
{
    FORBILD_Ellipt_Cyl ec;
    ec.rho = s.rho;
    ec.x = s.x;
    ec.y = s.y;
    ec.z = s.z;
    ec.l = s.l;
    ec.dx = s.dx;
    ec.dy = s.dy;
    ec.axis = s.axis;
    ec.a_x = s.a_x;
    ec.a_y = s.a_y;
    return ec;
}

__device__ FORBILD_Ellipt_Cyl_x _to_Ellipt_Cyl_x(const FORBILD_Shape s)
{
    FORBILD_Ellipt_Cyl_x ec;
    ec.rho = s.rho;
    ec.x = s.x;
    ec.y = s.y;
    ec.z = s.z;
    ec.l = s.l;
    ec.dy = s.dy;
    ec.dz = s.dz;
    return ec;
}

__device__ FORBILD_Ellipt_Cyl_y _to_Ellipt_Cyl_y(const FORBILD_Shape s)
{
    FORBILD_Ellipt_Cyl_y ec;
    ec.rho = s.rho;
    ec.x = s.x;
    ec.y = s.y;
    ec.z = s.z;
    ec.l = s.l;
    ec.dx = s.dx;
    ec.dz = s.dz;
    return ec;
}

__device__ FORBILD_Ellipt_Cyl_z _to_Ellipt_Cyl_z(const FORBILD_Shape s)
{
    FORBILD_Ellipt_Cyl_z ec;
    ec.rho = s.rho;
    ec.x = s.x;
    ec.y = s.y;
    ec.z = s.z;
    ec.l = s.l;
    ec.dx = s.dx;
    ec.dy = s.dy;
    return ec;
}

__device__ FORBILD_Sphere _to_Sphere(const FORBILD_Shape s)
{
    FORBILD_Sphere sphere;
    sphere.rho = s.rho;
    sphere.x = s.x;
    sphere.y = s.y;
    sphere.z = s.z;
    sphere.r = s.r;
    return sphere;
}

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
            return _in_region(v, _to_Box(shape));
        case 2:
            return _in_region(v, _to_Cone(shape));
        case 3:
            return _in_region(v, _to_Cone_x(shape));
        case 4:
            return _in_region(v, _to_Cone_y(shape));
        case 5:
            return _in_region(v, _to_Cone_z(shape));
        case 6:
            return _in_region(v, _to_Cylinder(shape));
        case 7:
            return _in_region(v, _to_Cylinder_x(shape));
        case 8:
            return _in_region(v, _to_Cylinder_y(shape));
        case 9:
            return _in_region(v, _to_Cylinder_z(shape));
        case 10:
            return _in_region(v, _to_Ellipsoid(shape));
        case 11:
            return _in_region(v, _to_Ellipsoid_free(shape));
        case 12:
            return _in_region(v, _to_Ellipt_Cyl(shape));
        case 13:
            return _in_region(v, _to_Sphere(shape));
        case 14:
            return _in_region(v, _to_Ellipt_Cyl_x(shape));
        case 15:
            return _in_region(v, _to_Ellipt_Cyl_y(shape));
        case 16:
            return _in_region(v, _to_Ellipt_Cyl_z(shape));
        default:
            return false;
    }
}

__device__ bbox get_bbox(
    FORBILD_Shape shape
)
{

    // To get a tighter bounding box, check the clipping planes here

    switch (shape.type) {
        case 0:
            // Should not occur
            assert(false);
        case 1:
            return get_bbox(_to_Box(shape));
        case 2:
            return get_bbox(_to_Cone(shape));
        case 3:
            return get_bbox(_to_Cone_x(shape));
        case 4:
            return get_bbox(_to_Cone_y(shape));
        case 5:
            return get_bbox(_to_Cone_z(shape));
        case 6:
            return get_bbox(_to_Cylinder(shape));
        case 7:
            return get_bbox(_to_Cylinder_x(shape));
        case 8:
            return get_bbox(_to_Cylinder_y(shape));
        case 9:
            return get_bbox(_to_Cylinder_z(shape));
        case 10:
            return get_bbox(_to_Ellipsoid(shape));
        case 11:
            return get_bbox(_to_Ellipsoid_free(shape));
        case 12:
            return get_bbox(_to_Ellipt_Cyl(shape));
        case 13:
            return get_bbox(_to_Sphere(shape));
        case 14:
            return get_bbox(_to_Ellipt_Cyl_x(shape));
        case 15:
            return get_bbox(_to_Ellipt_Cyl_y(shape));
        case 16:
            return get_bbox(_to_Ellipt_Cyl_z(shape));
        default:
            assert(false);
    }
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Shape shape
)
{
    // Return early if shape is null shape
    if (shape.type == 0) {
        return fvec2(0.f, 0.f);
    }

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
    maxf = max(minf, maxf);  // ensure maxf is greater than or equal to minf
    // Return early if interval is empty
    if (minf == maxf) {
        return fvec2(0.f, 0.f);
    }

    switch (shape.type) {
        case 0:  // Should already be handled above, but oh well
            return fvec2(0.f, 0.f);
        case 1:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Box(shape));
            break;
        case 2:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cone(shape));
            break;
        case 3:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cone_x(shape));
            break;
        case 4:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cone_y(shape));
            break;
        case 5:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cone_z(shape));
            break;
        case 6:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cylinder(shape));
            break;
        case 7:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cylinder_x(shape));
            break;
        case 8:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cylinder_y(shape));
            break;
        case 9:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Cylinder_z(shape));
            break;
        case 10:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipsoid(shape));
            break;
        case 11:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipsoid_free(shape));
            break;
        case 12:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipt_Cyl(shape));
            break;
        case 13:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Sphere(shape));
            break;
        case 14:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipt_Cyl_x(shape));
            break;
        case 15:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipt_Cyl_y(shape));
            break;
        case 16:
            interval = FORBILD_clip_line_segment(v1, v2, _to_Ellipt_Cyl_z(shape));
            break;
        default:
            interval = fvec2(0.f, 0.f);
            break;
    }
    minf = max(minf, interval.x);  // first  component
    maxf = min(maxf, interval.y);  // second component
    maxf = max(minf, maxf);  // ensure maxf is greater than or equal to minf
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
    return shape.rho;
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
__device__ float exact_line_integral_prioritise_last_recursive(
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
        val += exact_line_integral_prioritise_last_recursive(
            v1, v1 + dv*interval.x, shapes, new_num_shapes);
        val += exact_line_integral_prioritise_last_recursive(
            v1 + dv*interval.y, v2, shapes, new_num_shapes);
    }
    return val;
}

// A template method for all the shapes to use, including FORBILD_Shape
__device__ float exact_line_integral_prioritise_last(
        const fvec3                 v1,
        const fvec3                 v2,
        //Ts*                         shapes_in,
        //unsigned int                num_shapes_in
        fvec2*                      intervals_in,
        float*                      rhos_in,
        unsigned int                num_shapes_in
)
{
    // Base case
    if (num_shapes_in == 0) {
        return 0.f;
    }
    // Return if line segment has length 0
    const fvec3 dv = v2 - v1;
    const float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) {
        return 0.f;
    }

    fvec2* intervals = intervals_in;
    float* rhos = rhos_in;
    unsigned int num_shapes = num_shapes_in;

    // To avoid recursion, use a loop to find the first shape (beginning from the end)
    // which has any overlap with this line
    float rho;
    float progress = 0.f;
    float val = 0.f;
    while (progress < 1.f) {
        fvec2 interval = fvec2(1.f, 1.f);
        for (unsigned int i = 0; i < num_shapes; i++) {
            fvec2 interval_tmp = intervals[i];
            interval_tmp.x = max(interval_tmp.x, progress);
            interval_tmp.y = max(interval_tmp.y, progress);
            if (interval_tmp.y == interval_tmp.x) {
                if (i == 0) {
                    // Remove shape from beginning of list. Speeds up future loops.
                    intervals = &intervals[1];
                    rhos      = &rhos[1];
                    num_shapes--;
                    i--;  // This will cause i to underflow, but it's okay. It will be incremented back to 0.
                } else if (i == num_shapes-1) {
                    // Remove shape from end of list. Speeds up future loops.
                    num_shapes--;
                }
                continue;
            }
            if (interval_tmp.x <= interval.x) {
                rho = rhos[i];
                interval.x = interval_tmp.x;
                interval.y = interval_tmp.y;
            } else {
                // If a later shape overlaps, then shrink the domain of integration
                interval.y = min(interval.y, interval_tmp.x);
            }
        }
        float len = interval.y - interval.x;
        if (len > 0.f) {
            val += len * rho * dv_norm;
        }
        progress = interval.y;
    }
    return val;
}

// Commenting out this faster method because it introduces visible errors in the recons!
/*
// A template method for all the shapes to use, including FORBILD_Shape
template<typename Ts>
__device__ float exact_line_integral_prioritise_last_approx(
        const fvec3                 v1,
        const fvec3                 v2,
        Ts*                   shapes_in,
        unsigned int          num_shapes_in
)
{
    // Base case
    if (num_shapes_in == 0) {
        return 0.f;
    }
    // Return if line segment has length 0
    const fvec3 dv = v2 - v1;
    const float dv_norm = norm_fvec3(dv);
    if (dv_norm == 0) {
        return 0.f;
    }

    Ts* shapes = shapes_in;
    unsigned int num_shapes = num_shapes_in;

    // To avoid recursion, use a loop to find the first shape (beginning from the end)
    // which has any overlap with this line
    float val = 0.f;
    fvec3 vcurrent = v1;
    float dv_current_norm = dv_norm;
    float entry_intersection, exit_intersection;
    fvec2 interval;
    Ts shape_tmp;
    float rho;
    unsigned int num_shapes_consecutive_nonintersecting = 0;
    // An interval tolerance is needed. This is because when v1 is added to, it won't be exactly
    // equal to the intersection point identified, due to floating point error.
    // Maybe a better way to do this would be to have FORBILD_clip_line_segment return
    // the end points of the line. For now, I'll just include this negligible tolerance term.
    // NOTE: setting this tolerance to too small a number will greatly increase the time taken
    // to compute projections, because of this floating point error necessitating extra steps!
    // Setting it too low will also result in some bright spots in projections, where instability
    // from small numbers causes an overestimation of line integrals.
    // Example: 698ms --> 7856ms !
    // TODO: create a better solution that doesn't need fudging around floating-point error.
    // One option: add a "min_interval_start" parameter to the FORBILD_clip_line_segment function.
    // The quick pruning operations could happen based on v1 + (v2-v1)*min_interval_start,
    // and then the other calculations could happen based on the fixed input of v1, v2
    // which would not change throughout this loop. That keeps the end point of one line segment
    // exactly equal to the start point of the next line segment. This avoids the floating-point
    // instability problems. This approach worked previously, but that method was too slow
    // to be viable. It would need to be reimplemented by changing FORBILD_clip_line_segment
    // as suggested above.
    //const float interval_tolerance = 0.00001f;  // I tuned this amount. Do not set it lower.
    const float interval_tolerance = 0.0001f;
    unsigned int previous_shape_index = num_shapes;  // Initialise to a value not corresponding to any shape
    while (dv_current_norm > interval_tolerance) {
        num_shapes_consecutive_nonintersecting = 0;  // This will count up. If the loop ends with shape index 0 and a nonzero value for this number, then this many will be removed from the beginning of the array. This speeds up subsequent loops.
        entry_intersection = 1.1f;  // entry intersection. in [0,1], where 0 = vcurrent and 1 = v2. Set to >1 initially.
        exit_intersection = 1.2f;  // second lowest value found for the start of an intersection interval
        for (int i = num_shapes-1; i >= 0; i--) {

            shape_tmp = shapes[i];
            interval = FORBILD_clip_line_segment(vcurrent, v2, shape_tmp);
            if (interval.y <= interval.x + interval_tolerance) {
                // Empty interval.
                // No intersection with shape. Remove from consideration if possible
                if (i == num_shapes - 1) {
                    num_shapes --;  // remove shape from end
                } else {
                    num_shapes_consecutive_nonintersecting++;  // potentially remove shape from beginning
                }
            } else {
                // The interval to integrate over ends either naturally (end of shape, interval.y)
                // or it is interrupted by the start of a shape with a higher precedence (later in the list).
                // In the latter case, it will be the second-lowest interval.x found so far which is the stopping
                // point.

                num_shapes_consecutive_nonintersecting = 0;

                if (interval.x < entry_intersection) {
                    if (i == previous_shape_index) {
                        // To avoid an infinite loop caused by floating-point error,
                        // we need to explicitly skip integrating the same shape twice in a row.

                    } else {
                        // New entry intersection.
                        exit_intersection = min(entry_intersection, interval.y);  // stop the integral at the beginning of a higher precedence shape or the end of this shape; whichever is first
                        entry_intersection = interval.x;
                        previous_shape_index = i;
                        rho = FORBILD_get_rho(shape_tmp);
                        // If the intersection is immediate, then we don't need to search any further
                        if (entry_intersection <= interval_tolerance) {
                            entry_intersection = min(interval_tolerance/2.f, exit_intersection);
                            break;
                        }
                    }
                }
            }
            if ((i == 0) && (num_shapes_consecutive_nonintersecting > 0)) {
                // Shift up the array of shapes to skip the beginning in future loops
                shapes = &shapes[num_shapes_consecutive_nonintersecting];
                previous_shape_index -= num_shapes_consecutive_nonintersecting;
                num_shapes -= num_shapes_consecutive_nonintersecting;
            }
        }
        // If all shapes exhausted, then return.
        if (num_shapes == 0) { return val; }
        // If the program gets to this line, then there should be at least one intersection found.
        // At the end of the above loop, and getting to this line:
        //  - ...
        // Add the segment integral
        val += rho * (exit_intersection - entry_intersection) * dv_current_norm;
        if (exit_intersection >= 1.f) {
            // Return if integral is over.
            return val;
        }
        // Update the new starting point for further integration
        vcurrent = vcurrent + (v2 - vcurrent) * exit_intersection;
        dv_current_norm = norm_fvec3(v2 - vcurrent);
    }
    return val;
}
*/
