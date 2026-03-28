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
from .base import fvec3, uivec3

__all__ = [
    "config_cb_project",
    "vol_spec",
    "vol_tex_chunk_spec",
    "volume_chunk_spec",
]

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
