__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_z cone_z
)
{
    const fvec3 pos = fvec3(cone_z.x, cone_z.y, cone_z.z);
    const fvec3 v_rel = v - pos;
    if (abs(v_rel.z) > cone_z.l/2.f) {
        return false;
    }
    const float rad = cone_z.r1 + (v_rel.z/cone_z.l + 0.5f)*(cone_z.r2 - cone_z.r1);
    return (v_rel.x*v_rel.x + v_rel.y*v_rel.y <= rad*rad);
}

__device__ bbox get_bbox(
    FORBILD_Cone_z cone_z
)
{
    FORBILD_Cylinder_z cz;
    cz.x = cone_z.x;
    cz.y = cone_z.y;
    cz.z = cone_z.z;
    cz.r = max(cone_z.r1, cone_z.r2);
    cz.l = cone_z.l;
    return get_bbox(cz);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the shape.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_z cone_z
)
{

    // Handle the case where r2 = r1 here, as otherwise it causes
    // singularities in the floating point math
    if (cone_z.r1 == cone_z.r2) {
        FORBILD_Cylinder_z cylinder_z;
        cylinder_z.x = cone_z.x;
        cylinder_z.y = cone_z.y;
        cylinder_z.z = cone_z.z;
        cylinder_z.r = cone_z.r1;
        cylinder_z.l = cone_z.l;
        return FORBILD_clip_line_segment(v1, v2, cylinder_z);
    }

    const fvec3 pos = fvec3(cone_z.x, cone_z.y, cone_z.z);
    const fvec3 dv = v2 - v1;
    // To help handle unstable cases where the points are far away
    // from the cone, return early if the points are both above or both below the cone
    if (
        ((v1.z > pos.z+cone_z.l/2.f) && (v2.z > pos.z+cone_z.l/2.f)) ||
        ((v1.z < pos.z-cone_z.l/2.f) && (v2.z < pos.z-cone_z.l/2.f))
    ) {
        return fvec2(0.f, 0.f);
    }

    // To help handle unstable cases where the points are far away
    // from the cone, return early if the points are clearly
    // too far away to intersect with the cone
    // Interval to be in bounds of the cylinder radius
    // Will project v1, v2 into the plane orthogonal to cylinder z axis and containing cylinder pos
    fvec3 v1_projected = fvec3(v1.x, v1.y, pos.z);
    fvec3 v2_projected = fvec3(v2.x, v2.y, pos.z);
    fvec3 dv_projected = fvec3(dv.x, dv.y, pos.z);
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
    if (dist > max(cone_z.r1, cone_z.r2)) { return fvec2(0.f, 0.f); }

    const bool v1_in_region = _in_region(v1, cone_z);
    const bool v2_in_region = _in_region(v2, cone_z);
    if (v1_in_region && v2_in_region) {
        return fvec2(0.f, 1.f);
    }

    const fvec3 v1_rel = v1 - pos;
    float lam1, lam2;
    fvec2 interval = fvec2(0.f, 1.f);

    // First, handle the special case of dv points along z axis
    if ((dv.x == 0) && (dv.y == 0)) {
        // Equation for a non-truncated cone with axis parallel to the z axis:
        // x^2 + y^2 < r^2, where r depends on z.
        // x^2 + y^2 < (r1 + (r2-r1)*(z/l + 0.5))^2
        // |v_perp| < r1 + (r2-r1)*(vz/l + 0.5)
        //
        // when dv.x = dv.y = 0, then the intersection occurs when:
        // |v_perp| - r1 = (r2-r1)*(vz/l + 0.5)
        // (|v_perp| - r1)/(r2-r1) = vz/l + 0.5
        // vz = l*((|v_perp| - r1)/(r2-r1) - 0.5)
        const float zintersect_rel = cone_z.l * (
            (sqrt(v1_rel.x*v1_rel.x + v1_rel.y*v1_rel.y) - cone_z.r1)/(cone_z.r2 - cone_z.r1) - 0.5f);
        lam1 = (zintersect_rel - v1_rel.z)/dv.z;
        if ((cone_z.r1 > cone_z.r2) != (dv.z > 0)) {
            // upright cylinder and dv pointing down, or upside-down cylinder and dv pointing up
            interval.x = max(interval.x, lam1);
        } else {
            interval.y = min(interval.y, lam1);
        }
    } else {
        /// In this equation, 'v' is the vector relative to 'pos'
        // |v_perp| < r1 + (r2-r1)*(vz/l + 0.5)
        // ==> sqrt((v.x + lam*dv.x)^2 + (v.y + lam*dv.y)^2) = r1 + (r2-r1)*((v.z + lam*dv.z)/l + 0.5f)
        // ==> quadratic equation for lam, with 0 = a lam^2 + b lam + c with:
        const float r1 = cone_z.r1;
        const float r2 = cone_z.r2;
        const float rdiff = r2 - r1;
        const float l  = cone_z.l;
        const float vx = v1_rel.x;
        const float vy = v1_rel.y;
        const float vz = v1_rel.z;
        const float dx = dv.x;
        const float dy = dv.y;
        const float dz = dv.z;
        const float rdiff_on_l = rdiff/l;
        const double a = dx*dx + dy*dy - (dz*rdiff_on_l * dz*rdiff_on_l);
        //const float a = max(dx*dx,0.000001f) + max(dy*dy,0.000001f) - (dz*rdiff_on_l * dz*rdiff_on_l);
        const double b = (
            -(dz*rdiff/l)*(2*r1 + rdiff*(1 + 2*vz/l))
            + 2*(dx*vx + dy*vy)
        );
        const double c = (
            - r1*r1 - r1*rdiff - 0.25f*rdiff*rdiff + vx*vx + vy*vy
            - (vz*rdiff/l)*(2*r1 + rdiff*(1 + vz/l)) // no factor of 2 next to vz/l
        );
        // If a = 0, then the line has the same angle from the z axis as the cone's surface.
        if (a == 0) {
            // 0 = b lam + c
            if (b == 0) { // The line is exactly along the face of the cone
                return fvec2(0.f, 0.f);
            }
            // Along the line, the cylinder is present in an interval of one of the forms:
            // - [lam, infty)
            // - (-infty, lam]
            lam1 = -c/b;
            if ((cone_z.r1 > cone_z.r2) != (dv.z > 0)) {
                // upright cylinder and dv pointing down, or upside-down cylinder and dv pointing up
                interval.x = max(interval.x, lam1);
            } else {
                interval.y = min(interval.y, lam1);
            }
        } else {
            // Cases:
            // a < 0: line is closer to vertical than the cone angle
            // a > 0: line is further from vertical than the cone angle
            //const float disc = b*b - 4*a*c;// + 0.0001f;
            const float b_on_2a = b/a/2.;

            // Original code:
            //const float disc_on_4asq = b_on_2a*b_on_2a - (c/a);

            const float ac_on_bsq = a*c/(b*b);
            if (abs(ac_on_bsq) < 0.025f) {
                // This is probably unecessary, but it will improve on the floating point precision in some cases.
                //
                // Inspired by https://math.stackexchange.com/a/2007723,
                // (this post seems to indicate the unstable root as the stable one...)
                // we compute the stable root:
                // r_1 = (-b + sgn(b) sqrt(b*b - 4ac))/(2a)
                // Then use Vieta's formula for the product of roots, r_1 * r_2 = c/a
                // so r_2 = c/(a * r_1)
                // Note:
                // use            sqrt(b*b - 4ac) = sqrt(b^2) sqrt(1 - 4ac/b^2)
                // use            sqrt(b*b - 4ac) =  sgn(b) b sqrt(1 - 4ac/b^2)
                //    ---> sgn(b) sqrt(b*b - 4ac) =         b sqrt(1 - 4ac/b^2)
                //    ---> r_1 = ( -b + b sqrt(1 - 4ac/b^2)     ) / (2a)
                //    ---> r_1 = ( -b/(2a) ) * (1 - sqrt(1 - 4ac/(b^2)))
                // Then, I use the expansion
                // 1 - sqrt(1 - 4ac/(b^2)) ~ 2*( ac_on_bsq + ac_on_bsq^2 + 2*ac_on_bsq^3 )
                const float r_1 = -b_on_2a * 2*(
                    ac_on_bsq +
                    ac_on_bsq*ac_on_bsq +
                    2*ac_on_bsq*ac_on_bsq*ac_on_bsq
                );
                lam1 = r_1;
                lam2 = c / (a * r_1);
            } else {
                const float disc_on_4asq = max(0.f, b_on_2a*b_on_2a - (c/a));
                // WARNING:
                // I had previously returned fvec2(0.f, 0.f) when the disc_on_4asq was less than or equal to 0.
                // This is incorrect, since we may need to select everything outside this region, so we still need to follow the below code which branches based on the sign of a.
                const float pm_term_on_2a = sqrt(disc_on_4asq);
                lam1 = -b_on_2a - pm_term_on_2a;
                lam2 = -b_on_2a + pm_term_on_2a;
            }
            //const float disc_on_4asq = abs(b_on_2a*b_on_2a - (c/a));


            // Two possible cases:
            //  - the interval from (-b - pm_term)/(2a) to (-b + pm_term)/(2a)
            //  - everything excluding the above interval (think of a double-cone)
            // in the second case, we need to figure out which of the two remaining intervals
            // is the correct one for this cone.
            // The case is determined by the sign of a
            if (a > 0) {
                interval.x = max(interval.x, min(lam1, lam2));
                interval.y = min(interval.y, max(lam1, lam2));
            } else {
                if ((r1 > r2) != (dz > 0)) {
                    // upright cylinder and dv pointing down, or upside-down cylinder and dv pointing up
                    interval.x = max(interval.x, max(lam1, lam2));
                } else {
                    interval.y = min(interval.y, min(lam1, lam2));
                }
            }
        }
    }


    // Limit to the length/height of the cone
    if (dv.z != 0) {
        lam1 = (-0.5f * cone_z.l - v1_rel.z)/dv.z;
        lam2 = ( 0.5f * cone_z.l - v1_rel.z)/dv.z;
        interval.x = max(interval.x, min(lam1, lam2));
        interval.y = min(interval.y, max(lam1, lam2));
    }
    interval.y = max(interval.x, interval.y);
    return interval;
}
