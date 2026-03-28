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
    "fvec3",
    "uivec3",
]

uivec3 = kern_structs.KernelStruct(
    struct_name="uivec3",
    names=["x", "y", "z"],
    dtypes=[np.uint32, np.uint32, np.uint32],
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
)

fvec3 = kern_structs.KernelStruct(
    struct_name="fvec3",
    names=["x", "y", "z"],
    dtypes=[np.float32, np.float32, np.float32],
    force_dependencies=[
        uivec3
    ],  # used to define a constructor in terms of uivec3
    insert_into_struct="""
public:
    // implicit constructor from floats
    fvec3(float xi, float yi, float zi) : x(xi), y(yi), z(zi) {}
    // implicit constructor from a uivec3
    fvec3(uivec3 v) : x(v.x), y(v.y), z(v.z) {}
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


__device__ float dot_prod(
        fvec3 v1,
        fvec3 v2
)
{
    return v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
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
)
