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

"""Operation to apply on/betwen CUDA textures/surfaces."""

__all__ = [
    "fill",
    "invert",
    "multiply",
    "multiply_textures",
    "slice_nth_axis_to_3d_cupy_array",
    "softmask_to_cylinder_xy",
    "sum_along_nth_axis_to_2d_cupy_array",
]

import itertools
import math
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ...cuda import header_injection
from ...cuda.kern_structs import _np_dtype_to_c_type as _c_type_name_dict

if TYPE_CHECKING:
    from .texture import Texture

_BLOCKDIMS: tuple[int, int, int] = (10, 10, 10)

# See https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
# for information about native C++/CUDA datatypes.
# Here, I'm making corresponding numpy dtypes that I can use to pass in with CuPy. CuPy supports
# custom numpy dtypes. Otherwise, it is not possible to pass in arbitrary objects with CuPy.
_uint3 = np.dtype([("x", "<u4"), ("y", "<u4"), ("z", "<u4")])


def _shape_to_uint3(shape: tuple[int, int, int]) -> np.ndarray:
    return np.array(shape, dtype=_uint3)


def _get_module() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./texture_operations.cu",
        name_expressions=[
            "fill_surface<half>",
            "fill_surface<float>",
            "softmask_surface_to_cylinder_xy<half>",
            "softmask_surface_to_cylinder_xy<float>",
            "multiplicatively_invert_surface<half>",
            "multiplicatively_invert_surface<float>",
            "multiply_surface<half>",
            "multiply_surface<float>",
        ]
        + [
            f"multiply_surfaces<{t1},{t2},{t3}>"
            for (t1, t2, t3) in itertools.product(
                *itertools.repeat(["half", "float"], 3)
            )
        ]
        + [
            f"add_surfaces<{t1},{t2},{t3}>"
            for (t1, t2, t3) in itertools.product(
                *itertools.repeat(["half", "float"], 3)
            )
        ]
        + [f"slice_nth_axis_to_3d_array<{t1}>" for t1 in ["half", "float"]]
        + [
            f"sum_along_nth_axis_to_2d_array<{t1}>" for t1 in ["half", "float"]
        ],
    )
    module.compile()
    return module


_module: cp.RawModule = _get_module()


def _get_grid_block_dims(
    t: Texture,
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if len(t.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)
    shape: tuple[int, int, int] = t.shape  # ty:ignore[invalid-assignment]
    block_dims: tuple[int, int, int] = tuple(map(min, shape, _BLOCKDIMS))  # ty:ignore[invalid-assignment]
    grid_dims: tuple[int, int, int] = tuple(
        map(lambda s, b: math.ceil(s / b), shape, block_dims)
    )
    return grid_dims, block_dims


def _get_c_type_name(t: Texture) -> str:
    """E.g. convert 'np.float32' to 'float' and 'np.uint8' to 'unsigned short char'."""
    return _c_type_name_dict[t.dtype]


def fill(t: Texture, value: float) -> None:
    """Fill a Texture object with a constant value."""
    if not t.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.

    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"fill_surface<{ctype:s}>")
    # Check that it's a volume
    if len(t.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)

    val = np.array(value).astype(t.dtype)  # convert to the correct datatype
    args = (val, t.surfobj, _shape_to_uint3(t.shape))  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t), args)


def multiply(t: Texture, value: float) -> None:
    """Fill a Texture object with a constant value."""
    if not t.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.

    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"multiply_surface<{ctype:s}>")
    # Check that it's a volume
    if len(t.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)

    val = np.array(value).astype(t.dtype)  # convert to the correct datatype
    args = (val, t.surfobj, _shape_to_uint3(t.shape))  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t), args)


def multiply_textures(t1: Texture, t2: Texture, tout: Texture) -> None:
    """Fill a Texture object with a constant value."""
    if not tout.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.
    if not t1.is_surface_included or not t2.is_surface_included:
        msg = "One or more of the input textures does not include a surface."
        raise NotImplementedError(msg)

    # Check that it's a volume
    if len(t1.shape) != 3 or len(t2.shape) != 3 or len(tout.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)
    if t1.shape != t2.shape or t1.shape != tout.shape:
        msg = "Input textures do not have matching shape."
        raise ValueError(msg)

    ctypes = list(map(_get_c_type_name, (t1, t2, tout)))
    kern = _module.get_function(
        f"multiply_surfaces<{ctypes[0]:s},{ctypes[1]:s},{ctypes[2]:s}>"
    )

    args = (t1.surfobj, t2.surfobj, tout.surfobj, _shape_to_uint3(t1.shape))  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t1), args)


def add_textures(t1: Texture, t2: Texture, tout: Texture) -> None:
    """Fill a Texture object with a constant value."""
    if not tout.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.
    if not t1.is_surface_included or not t2.is_surface_included:
        msg = "One or more of the input textures does not include a surface."
        raise NotImplementedError(msg)

    # Check that it's a volume
    if len(t1.shape) != 3 or len(t2.shape) != 3 or len(tout.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)
    if t1.shape != t2.shape or t1.shape != tout.shape:
        msg = "Input textures do not have matching shape."
        raise ValueError(msg)

    ctypes = list(map(_get_c_type_name, (t1, t2, tout)))
    kern = _module.get_function(
        f"add_surfaces<{ctypes[0]:s},{ctypes[1]:s},{ctypes[2]:s}>"
    )

    args = (t1.surfobj, t2.surfobj, tout.surfobj, _shape_to_uint3(t1.shape))  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t1), args)


def invert(
    t: Texture,
    ignore_zeros: bool = False,
    softmax_epsilon: int | float | None = None,
) -> None:
    """Multiplicatively invert the values in the texture."""
    if not t.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.

    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"multiplicatively_invert_surface<{ctype:s}>")
    # Check that it's a volume
    if len(t.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)

    eps = np.float32(softmax_epsilon or 0)
    args = (t.surfobj, _shape_to_uint3(t.shape), np.bool(ignore_zeros), eps)  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t), args)


def softmask_to_cylinder_xy(t: Texture, relrad_start: float) -> None:
    """Fill a Texture object with a constant value."""
    if not t.is_surface_included:
        msg = "The input texture has no surface descriptor, and therefore cannot be written to."
        raise ValueError(msg)
    # No need for a texture to be present.
    if not 0 <= relrad_start <= 1:
        msg = "The input 'relrad_start' should be between 0 and 1."
        raise ValueError(msg)
    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"softmask_surface_to_cylinder_xy<{ctype:s}>")
    # Check that it's a volume
    if len(t.shape) != 3:
        msg = "Input texture was expected to be a volume, but it has an incorrect number of dimensions."
        raise ValueError(msg)

    rr_start = np.float32(relrad_start)  # convert to the correct datatype
    args = (rr_start, t.surfobj, _shape_to_uint3(t.shape))  # ty:ignore[invalid-argument-type]
    kern(*_get_grid_block_dims(t), args)


def slice_nth_axis_to_3d_cupy_array(  # noqa: C901
    t: Texture, out: cp.ndarray | None, axis: int, sl: slice
) -> cp.ndarray:
    """Copy a slice of the data from 2D-Layered or 3D texture into a 3D cupy array."""
    # Check validity of inputs
    if sl.step is not None:
        if sl.step != 1:
            msg = "Cannot use this method with a step size other than 1."
            raise NotImplementedError(msg)
    if not t.is_surface_included:
        msg = "No surface included in this texture. This is required to use this function."
        raise ValueError(msg)
    if len(t.shape) != 3:
        msg = "The provided texture is not three-dimensional."
        raise ValueError(msg)
    if axis not in (0, 1, 2):
        msg = "Invalid choice of axis. Choose 0, 1 or 2."
        raise ValueError(msg)

    # Regularise the slice so it has a definite start and stop index, and the stop index ends at the boundary if it goes over.
    sl_start = 0 if sl.start is None else sl.start
    sl_stop = t.shape[axis] if sl.stop is None else min(sl.stop, t.shape[axis])
    sl = slice(sl_start, sl_stop)
    sl_len = sl.stop - sl.start

    if sl_len <= 0:
        # return None
        msg = f"Slice is empty. Provided slice: {sl}. Array shape: {t.shape}"
        raise ValueError(msg)

    # Determine output shape
    match axis:
        case 0:
            out_shape = (sl_len, *t.shape[1:])
        case 1:
            out_shape = (t.shape[0], sl_len, t.shape[2])  # ty:ignore[index-out-of-bounds]
        case 2:
            out_shape = (*t.shape[:2], sl_len)
        case _:
            # This should never happen.
            msg = "This function has a bug."
            raise RuntimeError(msg)

    dtype = t.dtype
    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"slice_nth_axis_to_3d_array<{ctype:s}>")

    if out is None:
        out = cp.empty(out_shape, dtype=dtype, order="C")

    if out.shape != out_shape:
        msg = f"Output array is the wrong shape. Its shape is {out.shape} when it should be {out_shape}."
        raise ValueError(msg)

    if not out.flags["C_CONTIGUOUS"]:
        msg = "Cannot write to an array that isn't C contiguous at this time."
        raise NotImplementedError(msg)

    if out.dtype != dtype:
        msg = f"Wrong dtype for output array. Provided dtype: {out.dtype}. dtype corresponding to texture object: {dtype}."
        raise ValueError(msg)

    BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z = 10, 10, 10  # noqa: N806
    grid_dims = (
        math.ceil(out_shape[0] / BLOCKDIM_X),
        math.ceil(out_shape[1] / BLOCKDIM_Y),
        math.ceil(out_shape[2] / BLOCKDIM_Z),
    )
    block_dims = (BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z)

    args = (
        t.surfobj,
        *(np.uint32(l) for l in out_shape),
        np.uint32(sl_start),
        np.uint32(sl_stop),
        np.uint8(axis),
        *(np.uint64(l) for l in out.strides),
        np.bool_(t.is_flagset_layered),
        out,
    )

    kern(grid_dims, block_dims, args)
    return out


def sum_along_nth_axis_to_2d_cupy_array(  # noqa: C901
    t: Texture, out: cp.ndarray | None, axis: int
) -> cp.ndarray:
    """Sum the data from a 2D-Layered or 3D texture along an axis, into a 2D cupy array."""
    # Check validity of inputs
    if not t.is_surface_included:
        msg = "No surface included in this texture. This is required to use this function."
        raise ValueError(msg)
    if len(t.shape) != 3:
        msg = "The provided texture is not three-dimensional."
        raise ValueError(msg)
    if axis not in (0, 1, 2):
        msg = "Invalid choice of axis. Choose 0, 1 or 2."
        raise ValueError(msg)
    if len(t.shape) != 3:
        msg = "Input texture is not 3D."
        raise ValueError(msg)

    # Determine output shape
    match axis:
        case 0:
            out_shape = (t.shape[1], t.shape[2])  # ty:ignore[index-out-of-bounds]
        case 1:
            out_shape = (t.shape[0], t.shape[2])  # ty:ignore[index-out-of-bounds]
        case 2:
            out_shape = (t.shape[0], t.shape[1])
        case _:
            # This should never happen.
            msg = "This function has a bug."
            raise RuntimeError(msg)

    dtype = t.dtype
    ctype = _get_c_type_name(t)
    kern = _module.get_function(f"sum_along_nth_axis_to_2d_array<{ctype:s}>")

    if out is None:
        out = cp.zeros(out_shape, dtype=dtype, order="C")
    else:
        out.fill(
            0
        )  # We will add to the array, so we need to set it to zero first.

    if out.shape != out_shape:
        msg = f"Output array is the wrong shape. Its shape is {out.shape} when it should be {out_shape}."
        raise ValueError(msg)

    if not out.flags["C_CONTIGUOUS"]:
        msg = "Cannot write to an array that isn't C contiguous at this time."
        raise NotImplementedError(msg)

    if out.dtype != dtype:
        msg = f"Wrong dtype for output array. Provided dtype: {out.dtype}. dtype corresponding to texture object: {dtype}."
        raise ValueError(msg)

    BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z = 10, 10, 10  # noqa: N806
    grid_dims = (
        math.ceil(t.shape[0] / BLOCKDIM_X),
        math.ceil(t.shape[1] / BLOCKDIM_Y),
        math.ceil(t.shape[2] / BLOCKDIM_Z),  # ty:ignore[index-out-of-bounds]
    )
    block_dims = (BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z)

    args = (
        t.surfobj,
        *(np.uint32(l) for l in t.shape),
        np.uint8(axis),
        *(np.uint64(l) for l in out.strides),
        np.bool_(t.is_flagset_layered),
        out,
    )

    kern(grid_dims, block_dims, args)
    return out
