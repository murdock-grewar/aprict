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

config_cb_project = kern_structs.KernelStruct(
    struct_name="config_CB_project",
    names=[],
    dtypes=[],
)

volume_chunk_spec = kern_structs.KernelStruct(
    struct_name="volume_chunk_spec",
    names=[
        "idx_xstart",
        "idx_xstop",
        "idx_ystart",
        "idx_ystop",
        "idx_zstart",
        "idx_zstop",
    ],
    dtypes=[np.uint16, np.uint16, np.uint16, np.uint16, np.uint16, np.uint16],
)

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


det_spec = kern_structs.KernelStruct(
    struct_name="det_spec",
    names=[
        "pos",  # position of detector (nominally, the middle of it)
        "u",  # vector pointing along the x direction. Also determines pixel width.
        "v",  # vector pointing along the y direction. Also determines pixel height.
        "nu",
        "nv",  # number of columns/rows. i.e. length of x axis; length of y axis.
        "type",  # numeric integer describing the type of detector. Currently, leave it at 0 for a flat panel.
    ],
    dtypes=[fvec3, fvec3, fvec3, np.uint16, np.uint16, np.uint8],
)

proj_spec = kern_structs.KernelStruct(
    struct_name="proj_spec",
    names=["sourcepos", "dspec"],
    dtypes=[fvec3, det_spec],
)

vol_spec = kern_structs.KernelStruct(
    struct_name="vol_spec",
    names=[
        "ptr_000",  # pointer to the volume array, like cp.array(...).data.ptr
        # the pointer should point to what will be considered the arr[0][0][0] byte.
        # The pointer can be added to (like an ordinary integer) to get the correct offset
        # for wherever the relevant part of the volume starts
        "strides_bytes",  # strides of the 3 dimensions of the array, like cp.array(...).strides
        "shape",  # shape of the array. The underlying data may extend beyond this, but this shape
        # will determine how much is considered within range and should be accessed.
        "pos_000",  # x,y,z position of the center of the 0,0,0 voxel.
    ],
    dtypes=[np.uint64, uivec3, uivec3, fvec3],
)

vol_tex_chunk_spec = kern_structs.KernelStruct(
    struct_name="vol_tex_chunk_spec",
    names=[
        "offsets",  # unsigned integer offsets from the 0,0,0 base of the texture
        "shape",  # unsigned shape of the chunk
        "texpos",  # The 0,0,0 corner of the texture of which chunk is a part. NOTE: this is NOT the position of the center of the 0,0,0 voxel! It is the position of the corner of that voxel.
    ],
    dtypes=[uivec3, uivec3, fvec3],
)
