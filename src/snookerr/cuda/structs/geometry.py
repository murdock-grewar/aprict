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
"""Structures relating to projection/backprojection geometry."""

import numpy as np

from .. import kern_structs
from .base import fvec2, fvec3, uivec2, uivec3  # noqa: F401

__all__ = [
    "det_spec",
    "proj_chunk_spec_maxlen10",
    "proj_spec",
    "proj_tex_chunk_spec_maxlen10",
    "vol_spec",
    "vol_tex_chunk_spec",
    "volume_chunk_spec",
]


# ruff: disable[D101,N801]
class det_spec(
    kern_structs.KernelStruct,
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
):
    pass  # ty:ignore[invalid-argument-type]


class proj_spec(
    kern_structs.KernelStruct,
    struct_name="proj_spec",
    names=["sourcepos", "dspec"],
    dtypes=[fvec3, det_spec],
    force_dependencies=[
        fvec2,
    ],
    extra_code="""
/*
 * In order to mantain O(N^3) scaling in the backprojection, we need a finite, closed-form equation
 * which maps a voxel to its projection onto the detector, in terms of the detector indices.
 *
 * (The alterantive is to run a thread for each detector index, but the lack of atomic addition
 * for surfaces/textures leads to the need to use a lot of global memory atomicAdds for large
 * volumes, which is unacceptably slow. Maybe it's possible to make the memory accesses more
 * efficient for that case; I'm not sure.)
 */

__device__ fvec2 get_projected_detector_coords(
        const fvec3       point,
        const proj_spec   projspec
)
{
    if (projspec.dspec.type == 0) {
        /* Solve equation for p, lambda:
         * p . n = detector . n
         * p = sourcepos + lambda(point - sourcepos)
         * ==> detector . n = sourcepos . n + lambda (point - sourcepos) . n
         * ==> (detector - sourcepos) . n = lambda (point - sourcepos) . n
         * ==> lambda = [(detector - sourcepos) . n] / [(point - sourcepos) . n]
         * ==> p = sourcepos + lambda(point - sourcepos) with lambda as above
         */
        // Compute actual point of intersection, in 3D space
        const fvec3 n = cross_prod(projspec.dspec.u, projspec.dspec.v);
        fvec3 p = projspec.sourcepos + (point - projspec.sourcepos) * (
            dot_prod(projspec.dspec.pos - projspec.sourcepos, n) /
            dot_prod(point              - projspec.sourcepos, n)
        );
        // Compute the point's detector coordinates, from the center.
        p = p - projspec.dspec.pos;
        const float unorm = norm_fvec3(projspec.dspec.u);
        const float vnorm = norm_fvec3(projspec.dspec.v);
        return fvec2(
            dot_prod(projspec.dspec.u, p)/unorm/unorm,
            dot_prod(projspec.dspec.v, p)/vnorm/vnorm
        );
    } else {
        // UNIMPLEMENTED!!
        return fvec2(0.f, 0.f);
    }
}""",
):
    pass  # ty:ignore[invalid-argument-type]


class proj_chunk_spec_maxlen10(
    kern_structs.KernelStruct,
    struct_name="proj_chunk_spec_maxlen10",
    names=["projspecs", "numprojs", "projmemaddr", "projmemstrides"],
    dtypes=[
        kern_structs.KernelArray(dtype=proj_spec, shape=(10,)),
        np.uint8,
        np.uint64,
        uivec3,
    ],
):
    pass  # ty:ignore[invalid-argument-type]


class proj_tex_chunk_spec_maxlen10(
    kern_structs.KernelStruct,
    struct_name="proj_tex_chunk_spec_maxlen10",
    names=["projspecs", "numprojs", "tex_start_layer"],
    dtypes=[
        kern_structs.KernelArray(dtype=proj_spec, shape=(10,)),
        np.uint8,
        np.uint32,
    ],
):
    pass  # ty:ignore[invalid-argument-type]


# Note: this now used by the projection only.
# For the thing used by the backprojection, see 'vol_spec'. (This can also be used for chunks.)
class volume_chunk_spec(
    kern_structs.KernelStruct,
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
):
    pass  # ty:ignore[invalid-argument-type]


class vol_spec(
    kern_structs.KernelStruct,
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
):
    pass  # ty:ignore[invalid-argument-type]


class vol_tex_chunk_spec(
    kern_structs.KernelStruct,
    struct_name="vol_tex_chunk_spec",
    names=[
        "offsets",  # unsigned integer offsets from the 0,0,0 base of the texture
        "shape",  # unsigned shape of the chunk
        "texpos",  # The 0,0,0 corner of the texture of which chunk is a part. NOTE: this is NOT the position of the center of the 0,0,0 voxel! It is the position of the corner of that voxel.
    ],
    dtypes=[uivec3, uivec3, fvec3],
):
    pass  # ty:ignore[invalid-argument-type]


# ruff: enable[D101,N801]
