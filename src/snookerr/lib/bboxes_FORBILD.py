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
"""Tools to get bounding boxes of FORBILD phantoms/shapes."""

import itertools

import cupy as cp
import numpy as np

from ..cuda import header_injection
from ..cuda.kern_structs import _np_dtype_to_c_type as _c_type_name_dict
from ..cuda.structs import FORBILD_volumes as fv
from ..cuda.structs import bbox as ks_bbox

_BLOCKDIMS: tuple[int, int, int] = (
    300,
    1,
    1,
)  # around 300 FORBILD_Shape objects can fit in a shared memory block


def _get_module() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./get_bboxes_FORBILD.cu",
        name_expressions=[
            f"get_bboxes_FORBILD_shapes<{t2}>"
            for (t2,) in itertools.product(
                ["FORBILD_Shape"],
            )
        ],
    )
    # module.compile()
    return module


_module: cp.RawModule = _get_module()


def _get_c_type_name(arr: np.ndarray | cp.ndarray) -> str:
    """E.g. convert 'np.float32' to 'float' and 'np.uint8' to 'unsigned short char'."""
    return _c_type_name_dict[arr.dtype]


def bboxes_FORBILD_shapes(  # noqa: N802
    shapes: list[fv.Shape] | np.ndarray[np.dtype[fv.Shape]],  # ty:ignore[invalid-type-arguments]
    *,
    block: bool = True,
) -> cp.ndarray[tuple[int], np.dtype[ks_bbox]]:  # ty:ignore[invalid-type-arguments]
    """Get a cupy array of bounding boxes for all input shapes."""
    cp_shapes_arr = cp.array(shapes, dtype=fv.Shape)
    output_bboxes_arr = cp.zeros((cp_shapes_arr.size,), dtype=ks_bbox)
    kern = _module.get_function("get_bboxes_FORBILD_shapes<FORBILD_Shape>")
    args = (
        cp_shapes_arr,
        np.uint32(cp_shapes_arr.size),
        output_bboxes_arr,
    )
    grid_dims = (
        (cp_shapes_arr.size + (_BLOCKDIMS[0] - 1)) // _BLOCKDIMS[0],
        1,
        1,
    )
    kern(grid_dims, _BLOCKDIMS, args)
    if block:
        cp.cuda.get_current_stream().synchronize()
    return output_bboxes_arr


def overall_bbox(
    shapes: list[fv.Shape] | np.ndarray[np.dtype[fv.Shape]],  # ty:ignore[invalid-type-arguments]
) -> np.dtype[ks_bbox]:  # ty:ignore[invalid-type-arguments]
    """Return the overall bounding box encompassing all given shapes."""
    bboxes: np.ndarray[tuple[int], np.dtype[ks_bbox]] = bboxes_FORBILD_shapes(  # ty:ignore[invalid-type-arguments]
        shapes, block=True
    ).get()
    b = np.array(0, dtype=ks_bbox)
    b["xyz_min"]["x"] = np.min(bboxes[:]["xyz_min"]["x"], axis=0)
    b["xyz_min"]["y"] = np.min(bboxes[:]["xyz_min"]["y"], axis=0)
    b["xyz_min"]["z"] = np.min(bboxes[:]["xyz_min"]["z"], axis=0)
    b["xyz_max"]["x"] = np.max(bboxes[:]["xyz_max"]["x"], axis=0)
    b["xyz_max"]["y"] = np.max(bboxes[:]["xyz_max"]["y"], axis=0)
    b["xyz_max"]["z"] = np.max(bboxes[:]["xyz_max"]["z"], axis=0)
    return b  # ty:ignore[invalid-return-type]
