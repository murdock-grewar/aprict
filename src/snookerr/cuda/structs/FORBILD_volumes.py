# noqa: N999
# Copyright (C) 2026  Murdock Grewar
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""Structures encoding FORBILD primitives and other related things."""

import numpy as np

from .. import kern_structs
from .base import bbox, fvec2, fvec3, uivec2, uivec3  # noqa: F401

__all__ = [
    "Box",
    "ClipAxisSpec",
    "Cone",
    "Cone_x",
    "Cone_y",
    "Cone_z",
    "Cylinder",
    "Cylinder_x",
    "Cylinder_y",
    "Cylinder_z",
    "Ellipsoid",
    "Ellipsoid_free",
    "Ellipt_Cyl",
    "Ellipt_Cyl_x",
    "Ellipt_Cyl_y",
    "Ellipt_Cyl_z",
    "Shape",
    "Sphere",
]


# ruff: disable[D101,N801]
class Sphere(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Sphere",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
    ],
    dtypes=[np.float32, np.float32, np.float32, np.float32, np.float32],
    force_dependencies=[fvec2, fvec3, bbox],
    extra_code="""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Box(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Box",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",  # side length along x axis
        "dy",
        "dz",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[fvec2, fvec3, bbox],
    extra_code=r"""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cylinder(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
        "axis",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
    ],
    force_dependencies=[fvec2, fvec3, bbox],
    extra_code="""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cylinder_x(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],
    extra_code=r"""
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

__device__ bbox get_bbox(
    FORBILD_Cylinder_x c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.l/2.f;
    b.xyz_max.x = c.x + c.l/2.f;
    b.xyz_min.y = c.y - c.r;
    b.xyz_max.y = c.y + c.r;
    b.xyz_min.z = c.z - c.r;
    b.xyz_max.z = c.z + c.r;
    return b;
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cylinder_y(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],
    extra_code=r"""
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

__device__ bbox get_bbox(
    FORBILD_Cylinder_y c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.r;
    b.xyz_max.x = c.x + c.r;
    b.xyz_min.y = c.y - c.l/2.f;
    b.xyz_max.y = c.y + c.l/2.f;
    b.xyz_min.z = c.z - c.r;
    b.xyz_max.z = c.z + c.r;
    return b;
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cylinder_z(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cylinder_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "r",
        "l",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cylinder, fvec2, fvec3, bbox],
    extra_code=r"""
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

__device__ bbox get_bbox(
    FORBILD_Cylinder_z c
)
{
    bbox b;
    b.xyz_min.x = c.x - c.r;
    b.xyz_max.x = c.x + c.r;
    b.xyz_min.y = c.y - c.r;
    b.xyz_max.y = c.y + c.r;
    b.xyz_min.z = c.z - c.l/2.f;
    b.xyz_max.z = c.z + c.l/2.f;
    return b;
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipsoid(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipsoid",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",  # half-axis length along x axis
        "dy",
        "dz",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Sphere, fvec2, fvec3, bbox],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid ellipsoid
)
{
    const fvec3 pos = fvec3(ellipsoid.x, ellipsoid.y, ellipsoid.z);
    fvec3 v_rel = v - pos;
    v_rel.x /= ellipsoid.dx;
    v_rel.y /= ellipsoid.dy;
    v_rel.z /= ellipsoid.dz;
    return (norm_fvec3(v_rel) <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipsoid e
)
{
    bbox b;
    b.xyz_min.x = e.x - e.dx;
    b.xyz_max.x = e.x + e.dx;
    b.xyz_min.y = e.y - e.dy;
    b.xyz_max.y = e.y + e.dy;
    b.xyz_min.z = e.z - e.dz;
    b.xyz_max.z = e.z + e.dz;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid ellipsoid
)
{
    const fvec3 pos = fvec3(ellipsoid.x, ellipsoid.y, ellipsoid.z);
    fvec3 v1_rel = v1 - pos;
    v1_rel.x /= ellipsoid.dx;
    v1_rel.y /= ellipsoid.dy;
    v1_rel.z /= ellipsoid.dz;
    fvec3 v2_rel = v2 - pos;
    v2_rel.x /= ellipsoid.dx;
    v2_rel.y /= ellipsoid.dy;
    v2_rel.z /= ellipsoid.dz;
    FORBILD_Sphere sphere;
    sphere.x = 0.f;
    sphere.y = 0.f;
    sphere.z = 0.f;
    sphere.r = 1.f;
    return FORBILD_clip_line_segment(v1_rel, v2_rel, sphere);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipsoid_free(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipsoid_free",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "a_x",  # these are meant to be three orthogonal vectors. There's no enforcement of it, though.
        "a_y",
        "a_z",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
        fvec3,
        fvec3,
    ],
    force_dependencies=[Sphere, fvec2, fvec3, bbox],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipsoid_free ef
)
{
    const fvec3 pos = fvec3(ef.x, ef.y, ef.z);
    fvec3 v_rel = v - pos;
    fvec3 ax_normalised = ef.a_x / norm_fvec3(ef.a_x);
    float vx = dot_prod(v_rel, ax_normalised);
    fvec3 ay_normalised = ef.a_y / norm_fvec3(ef.a_y);
    float vy = dot_prod(v_rel, ay_normalised);
    fvec3 az_normalised = ef.a_z / norm_fvec3(ef.a_z);
    float vz = dot_prod(v_rel, az_normalised);
    vx /= ef.dx;
    vy /= ef.dy;
    vz /= ef.dz;
    return (vx*vx + vy*vy + vz*vz <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipsoid_free ef
)
{
    fvec3 ax = ef.a_x / norm_fvec3(ef.a_x);
    fvec3 ay = ef.a_y / norm_fvec3(ef.a_y);
    fvec3 az = ef.a_z / norm_fvec3(ef.a_z);
    bbox b;
    float d = abs(ef.dx*ax.x) + abs(ef.dy*ay.x) + abs(ef.dz*az.x);
    b.xyz_min.x = ef.x - d;
    b.xyz_max.x = ef.x + d;
    d = abs(ef.dx*ax.y) + abs(ef.dy*ay.y) + abs(ef.dz*az.y);
    b.xyz_min.y = ef.y - d;
    b.xyz_max.y = ef.y + d;
    d = abs(ef.dx*ax.z) + abs(ef.dy*ay.z) + abs(ef.dz*az.z);
    b.xyz_min.z = ef.z - d;
    b.xyz_max.z = ef.z + d;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipsoid_free ef
)
{
    const fvec3 pos = fvec3(ef.x, ef.y, ef.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 ax_normalised = ef.a_x / norm_fvec3(ef.a_x);
    fvec3 ay_normalised = ef.a_y / norm_fvec3(ef.a_y);
    fvec3 az_normalised = ef.a_z / norm_fvec3(ef.a_z);
    float v1x = dot_prod(v1_rel, ax_normalised);
    float v1y = dot_prod(v1_rel, ay_normalised);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v2x = dot_prod(v2_rel, ax_normalised);
    float v2y = dot_prod(v2_rel, ay_normalised);
    float v2z = dot_prod(v2_rel, az_normalised);
    v1x /= ef.dx;
    v1y /= ef.dy;
    v1z /= ef.dz;
    v2x /= ef.dx;
    v2y /= ef.dy;
    v2z /= ef.dz;
    FORBILD_Sphere sphere;
    sphere.x = 0.f;
    sphere.y = 0.f;
    sphere.z = 0.f;
    sphere.r = 1.f;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), sphere);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipt_Cyl(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dy",  # half-axis length
        # These vectors should be orthogonal, but there's no enforcement of that:
        "axis",  # direction of the principal direction.
        "a_x",  # direction of an orthogonal direction.
        "a_y",  # direction of an orthogonal direction.
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
        fvec3,
        fvec3,
    ],
    force_dependencies=[Cylinder_z, fvec2, fvec3],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Ellipt_Cyl ecyl
)
{
    const fvec3 pos = fvec3(ecyl.x, ecyl.y, ecyl.z);
    fvec3 v_rel = v - pos;
    fvec3 az_normalised = ecyl.axis / norm_fvec3(ecyl.axis);
    float vz = dot_prod(v_rel, az_normalised);
    if (abs(vz) > ecyl.l/2.f) {
        return false;
    }

    fvec3 ax_normalised = ecyl.a_x / norm_fvec3(ecyl.a_x);
    float vx = dot_prod(v_rel, ax_normalised);
    fvec3 ay_normalised = ecyl.a_y / norm_fvec3(ecyl.a_y);
    float vy = dot_prod(v_rel, ay_normalised);
    vx /= ecyl.dx;
    vy /= ecyl.dy;
    return (vx*vx + vy*vy <= 1);
}

__device__ bbox get_bbox(
    FORBILD_Ellipt_Cyl ecyl
)
{
    fvec3 ax = ecyl.a_x / norm_fvec3(ecyl.a_x);
    fvec3 ay = ecyl.a_y / norm_fvec3(ecyl.a_y);
    fvec3 az = ecyl.axis / norm_fvec3(ecyl.axis);
    bbox b;
    float d = abs(ecyl.dx*ax.x) + abs(ecyl.dy*ay.x) + abs(ecyl.l/2.f*az.x);
    b.xyz_min.x = ecyl.x - d;
    b.xyz_max.x = ecyl.x + d;
    d = abs(ecyl.dx*ax.y) + abs(ecyl.dy*ay.y) + abs(ecyl.l/2.f*az.y);
    b.xyz_min.y = ecyl.y - d;
    b.xyz_max.y = ecyl.y + d;
    d = abs(ecyl.dx*ax.z) + abs(ecyl.dy*ay.z) + abs(ecyl.l/2.f*az.z);
    b.xyz_min.z = ecyl.z - d;
    b.xyz_max.z = ecyl.z + d;
    return b;
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Ellipt_Cyl ecyl
)
{
    const fvec3 pos = fvec3(ecyl.x, ecyl.y, ecyl.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 ax_normalised = ecyl.a_x / norm_fvec3(ecyl.a_x);
    fvec3 ay_normalised = ecyl.a_y / norm_fvec3(ecyl.a_y);
    fvec3 az_normalised = ecyl.axis / norm_fvec3(ecyl.axis);
    float v1x = dot_prod(v1_rel, ax_normalised);
    float v1y = dot_prod(v1_rel, ay_normalised);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v2x = dot_prod(v2_rel, ax_normalised);
    float v2y = dot_prod(v2_rel, ay_normalised);
    float v2z = dot_prod(v2_rel, az_normalised);
    v1x /= ecyl.dx;
    v1y /= ecyl.dy;
    v2x /= ecyl.dx;
    v2y /= ecyl.dy;
    FORBILD_Cylinder_z cylinder_z;
    cylinder_z.x = 0.f;
    cylinder_z.y = 0.f;
    cylinder_z.z = 0.f;
    cylinder_z.r = 1.f;
    cylinder_z.l = ecyl.l;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), cylinder_z);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipt_Cyl_x(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dy",  # half-axis length
        "dz",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],
    extra_code="""
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
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipt_Cyl_y(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dz",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],
    extra_code="""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Ellipt_Cyl_z(  # Cylinder with an elliptical cross-section
    kern_structs.KernelStruct,
    struct_name="FORBILD_Ellipt_Cyl_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",  # length along principal axis
        "dx",  # half-axis length
        "dy",  # half-axis length
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Ellipt_Cyl, fvec2, fvec3, bbox],
    extra_code="""
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
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


# Other cone cases may be reduced to this case
class Cone_z(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_z",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",  # radius at z = cone.z - cone.l/2
        "r2",  # radius at z = cone.z + cone.l/2
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cylinder_z, fvec2, fvec3, bbox],
    extra_code=r"""
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
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
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
        const float a = dx*dx + dy*dy - (dz*rdiff/l * dz*rdiff/l);
        const float b = (
            -(dz*rdiff/l)*(2*r1 + rdiff*(1 + 2*vz/l))
            + 2*(dx*vx + dy*vy)
        );
        const float c = (
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
            const float disc = b*b - 4*a*c;
            if (disc <= 0) {  // no intersection
                return fvec2(0.f, 0.f);
            }
            const float pm_term = sqrt(disc);
            // Two possible cases:
            //  - the interval from (-b - pm_term)/(2a) to (-b + pm_term)/(2a)
            //  - everything excluding the above interval (think of a double-cone)
            // in the second case, we need to figure out which of the two remaining intervals
            // is the correct one for this cone.
            // The case is determined by the sign of a
            lam1 = (-b - pm_term)/(2*a);
            lam2 = (-b + pm_term)/(2*a);
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


"""Cone
truncated cone, arbitrary orientation in space as given by the axis
axis(expression,expression,expression). Length l, radii r1 and r2 at the two ends of the truncated cone. Moving along the cone axis in the direction given by axis, first the end corresponding to r1, then the end corresponding to r2 is met. x, y and z define the center of the truncated cone axis."""


class Cone(
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
        "axis",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        fvec3,
    ],
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder, bbox],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone cone
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone.x, cone.y, cone.z);
    fvec3 v_rel = v - pos;
    fvec3 az_normalised = cone.axis / norm_fvec3(cone.axis);
    float vz = dot_prod(v_rel, az_normalised);
    float vx = sqrt(max(0.f, (v_rel.x*v_rel.x + v_rel.y*v_rel.y + v_rel.z*v_rel.z) - vz*vz));
    float vy = 0.f;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone.l;
    cz.r1 = cone.r1;
    cz.r2 = cone.r2;
    return _in_region(fvec3(vx, vy, vz), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone cone
)
{
    FORBILD_Cylinder c;
    c.x = cone.x;
    c.y = cone.y;
    c.z = cone.z;
    c.r = max(cone.r1, cone.r2);
    c.l = cone.l;
    c.axis = cone.axis;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone cone
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone.x, cone.y, cone.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    fvec3 az_normalised = cone.axis / norm_fvec3(cone.axis);
    float v1z = dot_prod(v1_rel, az_normalised);
    float v1x = sqrt(max(0.f, (v1_rel.x*v1_rel.x + v1_rel.y*v1_rel.y + v1_rel.z*v1_rel.z) - v1z*v1z));
    float v1y = 0.f;
    float v2z = dot_prod(v2_rel, az_normalised);
    float v2x, v2y;
    if (v1x == 0) {
        v2x = sqrt(max(0.f, (v2_rel.x*v2_rel.x + v2_rel.y*v2_rel.y + v2_rel.z*v2_rel.z) - v2z*v2z));
        v2y = 0.f;
    } else {
        fvec3 xhat = v1_rel - az_normalised*v1z;
        xhat = xhat / norm_fvec3(xhat);
        v2x = dot_prod(v2_rel, xhat);
        v2y = sqrt(max(0.f, (v2_rel.x*v2_rel.x + v2_rel.y*v2_rel.y + v2_rel.z*v2_rel.z) - v2x*v2x - v2z*v2z));
    }
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone.l;
    cz.r1 = cone.r1;
    cz.r2 = cone.r2;
    return FORBILD_clip_line_segment(fvec3(v1x, v1y, v1z), fvec3(v2x, v2y, v2z), cz);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cone_x(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_x",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder_x, bbox],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_x cone_x
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_x.x, cone_x.y, cone_x.z);
    fvec3 v_rel = v - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_x.l;
    cz.r1 = cone_x.r1;
    cz.r2 = cone_x.r2;
    return _in_region(fvec3(v_rel.z, v_rel.y, v_rel.x), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone_x cone_x
)
{
    FORBILD_Cylinder_x c;
    c.x = cone_x.x;
    c.y = cone_x.y;
    c.z = cone_x.z;
    c.r = max(cone_x.r1, cone_x.r2);
    c.l = cone_x.l;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_x cone_x
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_x.x, cone_x.y, cone_x.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_x.l;
    cz.r1 = cone_x.r1;
    cz.r2 = cone_x.r2;
    return FORBILD_clip_line_segment(fvec3(v1_rel.z, v1_rel.y, v1_rel.x), fvec3(v2_rel.z, v2_rel.y, v2_rel.x), cz);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class Cone_y(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_Cone_y",
    names=[
        "rho",
        "x",
        "y",
        "z",
        "l",
        "r1",
        "r2",
    ],
    dtypes=[
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
        np.float32,
    ],
    force_dependencies=[Cone_z, fvec2, fvec3, Cylinder_y, bbox],
    extra_code="""
__device__ bool _in_region(
    fvec3 v,
    FORBILD_Cone_y cone_y
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_y.x, cone_y.y, cone_y.z);
    fvec3 v_rel = v - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_y.l;
    cz.r1 = cone_y.r1;
    cz.r2 = cone_y.r2;
    return _in_region(fvec3(v_rel.x, v_rel.z, v_rel.y), cz);
}

__device__ bbox get_bbox(
    FORBILD_Cone_y cone_y
)
{
    FORBILD_Cylinder_y c;
    c.x = cone_y.x;
    c.y = cone_y.y;
    c.z = cone_y.z;
    c.r = max(cone_y.r1, cone_y.r2);
    c.l = cone_y.l;
    return get_bbox(c);
}

__device__ fvec2 FORBILD_clip_line_segment(
    // Will output a subinterval of [0,1] specifying the portion of the line from v1 to v2 that is contained within the plane.
    fvec3 v1,
    fvec3 v2,
    FORBILD_Cone_y cone_y
)
{
    // Refer to the Cone_z case
    const fvec3 pos = fvec3(cone_y.x, cone_y.y, cone_y.z);
    fvec3 v1_rel = v1 - pos;
    fvec3 v2_rel = v2 - pos;
    FORBILD_Cone_z cz;
    cz.x = 0.f;
    cz.y = 0.f;
    cz.z = 0.f;
    cz.l  = cone_y.l;
    cz.r1 = cone_y.r1;
    cz.r2 = cone_y.r2;
    return FORBILD_clip_line_segment(fvec3(v1_rel.x, v1_rel.z, v1_rel.y), fvec3(v2_rel.x, v2_rel.z, v2_rel.y), cz);
}
""",
):
    pass  # ty:ignore[invalid-argument-type]


class ClipAxisSpec(  # Cone with axis parallel to x axis
    kern_structs.KernelStruct,
    struct_name="FORBILD_ClipAxisSpec",
    names=[
        "axis",  # note: it should be of unit length. This is NOT enforced!
        "dotprod",
        "mode",  # 0 = don't apply this plane. 1 --> enforce axis.r <= dotprod. 2 --> enforce axis.r > dotprod
    ],
    dtypes=[fvec3, np.float32, np.uint8],
    force_dependencies=[fvec2, fvec3],
    extra_code="""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


_dict_shape_name_to_type_num: dict[str, int] = {
    "Box": 1,
    "Cone": 2,
    "Cone_x": 3,
    "Cone_y": 4,
    "Cone_z": 5,
    "Cylinder": 6,
    "Cylinder_x": 7,
    "Cylinder_y": 8,
    "Cylinder_z": 9,
    "Ellipsoid": 10,
    "Ellipsoid_free": 11,
    "Ellipt_Cyl": 12,
    "Sphere": 13,
    "Ellipt_Cyl_x": 14,
    "Ellipt_Cyl_y": 15,
    "Ellipt_Cyl_z": 16,
}


class Shape(  # A generic container for any shape
    kern_structs.KernelStruct,
    struct_name="FORBILD_Shape",
    names=[
        "type",  # specifies which shape this is. 0 is reserved for "nothing", i.e. empty shape
        "rho",
        # up to three clipping planes
        "cas1",
        "cas2",
        "cas3",
        # List all fields which can occur in any shape
        "x",
        "y",
        "z",
        "dx",
        "dy",
        "dz",
        "r",
        "r1",
        "r2",
        "l",
        "a_x",
        "a_y",
        "a_z",
        "axis",
    ],
    dtypes=[
        np.uint8,  # type. value 0 makes this an non-existent shape
        np.float32,  # rho
        ClipAxisSpec,
        ClipAxisSpec,
        ClipAxisSpec,
        np.float32,  # x
        np.float32,  # y
        np.float32,  # z
        np.float32,  # dx
        np.float32,  # dy
        np.float32,  # dz
        np.float32,  # r
        np.float32,  # r1
        np.float32,  # r2
        np.float32,  # l
        fvec3,  # a_x
        fvec3,  # a_y
        fvec3,  # a_z
        fvec3,  # axis
    ],
    force_dependencies=[
        fvec2,
        fvec3,
        bbox,
        Box,
        Cone,
        Cone_x,
        Cone_y,
        Cone_z,
        Cylinder,
        Cylinder_x,
        Cylinder_y,
        Cylinder_z,
        Ellipsoid,
        Ellipsoid_free,
        Ellipt_Cyl,
        Ellipt_Cyl_x,
        Ellipt_Cyl_y,
        Ellipt_Cyl_z,
        Sphere,
    ],
    extra_code=r"""
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
""",
):
    pass  # ty:ignore[invalid-argument-type]


# ruff: enable[D101,N801]
