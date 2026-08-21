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
"""A struct used to configure the CB projection kernel."""

import numpy as np

from .. import kern_structs

__all__ = [
    "bbox",
    "fvec2",
    "fvec3",
    "uivec2",
    "uivec3",
    "ulvec3",
]


# ruff: disable[undocumented-public-class,invalid-class-name]
class uivec2(
    kern_structs.KernelStruct,
    struct_name="uivec2",
    names=["x", "y"],
    dtypes=[np.uint32, np.uint32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor from unsigned ints
    __device__ uivec2(unsigned int xi, unsigned int yi) : x(xi), y(yi) {}
    // default constructor, used with i.e. 'uivec2 v;'
    uivec2() = default;
""",
):
    pass


class fvec2(
    kern_structs.KernelStruct,
    struct_name="fvec2",
    names=["x", "y"],
    dtypes=[np.float32, np.float32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor from floats
    __device__ fvec2(float xi, float yi) : x(xi), y(yi) {}
    // default constructor, used with i.e. 'fvec2 v;'
    fvec2() = default;
""",
):
    pass


class uivec3(
    kern_structs.KernelStruct,
    struct_name="uivec3",
    names=["x", "y", "z"],
    dtypes=[np.uint32, np.uint32, np.uint32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor from floats
    __device__ uivec3(unsigned int xi, unsigned int yi, unsigned int zi) : x(xi), y(yi), z(zi) {}
    // default constructor, used with i.e. 'uivec3 v;'
    uivec3() = default;
""",
    extra_code="""
__device__ uivec3 operator+(
        uivec3 v1,
        uivec3 v2
)
{
    uivec3 v3;
    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
    return v3;
}


__device__ uivec3 operator-(
        uivec3 v1,
        uivec3 v2
)
{
    uivec3 v3;
    v3.x = v1.x - v2.x;
    v3.y = v1.y - v2.y;
    v3.z = v1.z - v2.z;
    return v3;
}


__device__ float dot_prod(
        uivec3 v1,
        uivec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

""",
):
    pass


class ulvec3(
    kern_structs.KernelStruct,
    struct_name="ulvec3",
    names=["x", "y", "z"],
    dtypes=[np.uint64, np.uint64, np.uint64],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor
    __device__ ulvec3(unsigned long xi, unsigned long yi, unsigned long zi) : x(xi), y(yi), z(zi) {}
    // default constructor, used with i.e. 'ulvec3 v;'
    ulvec3() = default;
""",
):
    pass


class fvec3(
    kern_structs.KernelStruct,
    struct_name="fvec3",
    names=["x", "y", "z"],
    dtypes=[np.float32, np.float32, np.float32],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    force_dependencies=[
        uivec3
    ],  # used to define a constructor in terms of uivec3  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor from floats
    __device__ fvec3(float xi, float yi, float zi) : x(xi), y(yi), z(zi) {}
    // implicit constructor from a uivec3
    __device__ fvec3(uivec3 v) : x(v.x), y(v.y), z(v.z) {}
    // default constructor, used with i.e. 'fvec3 v;'
    fvec3() = default;
""",
    extra_code="""
__device__ fvec3 operator+(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x + v2.x;
    v3.y = v1.y + v2.y;
    v3.z = v1.z + v2.z;
    return v3;
}


__device__ fvec3 operator-(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x - v2.x;
    v3.y = v1.y - v2.y;
    v3.z = v1.z - v2.z;
    return v3;
}


__device__ fvec3 operator*(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x * v2.x;
    v3.y = v1.y * v2.y;
    v3.z = v1.z * v2.z;
    return v3;
}


__device__ fvec3 operator/(
        fvec3 v1,
        fvec3 v2
)
{
    fvec3 v3;
    v3.x = v1.x / v2.x;
    v3.y = v1.y / v2.y;
    v3.z = v1.z / v2.z;
    return v3;
}


__device__ float dot_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
}

__device__ fvec3 cross_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return fvec3(
        v1.y*v2.z - v1.z*v2.y,
        v1.z*v2.x - v1.x*v2.z,
        v1.x*v2.y - v1.y*v2.x
    );
}

__device__ float min_of_components(
        fvec3 v
)
{
    return min(v.x, min(v.y, v.z));
}


__device__ float max_of_components(
        fvec3 v
)
{
    return max(v.x, max(v.y, v.z));
}


__device__ float norm_fvec3(
        fvec3 v
)
{
    return sqrt(dot_prod(v,v));
}


__device__ fvec3 operator*(
        fvec3 v,
        float f
)
{
    fvec3 o;
    o.x = v.x * f;
    o.y = v.y * f;
    o.z = v.z * f;
    return o;
}


__device__ fvec3 operator/(
        fvec3 v,
        float f
)
{
    fvec3 o;
    o.x = v.x / f;
    o.y = v.y / f;
    o.z = v.z / f;
    return o;
}
""",
):
    pass


class bbox(
    kern_structs.KernelStruct,
    struct_name="bbox",
    names=["xyz_min", "xyz_max"],
    dtypes=[fvec3, fvec3],  # ty: ignore[invalid-argument-type]  # ty is bugged: see https://github.com/astral-sh/ty/issues/3198
    insert_into_struct="""
public:
    // implicit constructor from fvec3s
    __device__ bbox(fvec3 xyz_mini, fvec3 xyz_maxi) : xyz_min(xyz_mini), xyz_max(xyz_maxi) {}
    // default constructor, used with i.e. 'bbox b;'
    bbox() = default;
""",
):
    pass


# ruff: enable[undocumented-public-class,invalid-class-name]
