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
"""Tools to sample FORBILD phantoms."""

import itertools
import math

import cupy as cp
import numpy as np

from ..cuda import header_injection
from ..cuda.kern_structs import _np_dtype_to_c_type as _c_type_name_dict
from ..cuda.structs import FORBILD_volumes as fv

_BLOCKDIMS: tuple[int, int, int] = (10, 10, 10)

# See https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html
# for information about native C++/CUDA datatypes.
# Here, I'm making corresponding numpy dtypes that I can use to pass in with CuPy. CuPy supports
# custom numpy dtypes. Otherwise, it is not possible to pass in arbitrary objects with CuPy.
_uint3 = np.dtype([("x", "<u4"), ("y", "<u4"), ("z", "<u4")])
_ulong3 = np.dtype([("x", "<u8"), ("y", "<u8"), ("z", "<u8")])


def _tuple_to_uint3(tup: tuple[int, int, int]) -> np.ndarray:
    return np.array(tup, dtype=_uint3)


def _tuple_to_ulong3(tup: tuple[int, int, int]) -> np.ndarray:
    return np.array(tup, dtype=_ulong3)


def _get_module() -> cp.RawModule:
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./sample_FORBILD_volumes.cu",
        name_expressions=[
            f"add_FORBILD_sampling<{t1},{t2}>"
            for (t1, t2) in itertools.product(
                ["half", "float", "double"],
                ["FORBILD_Shape"],
            )
        ],
    )
    # module.compile()
    return module


_module: cp.RawModule = _get_module()


def _get_grid_block_dims(
    vol_shape: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if len(vol_shape) != 3:
        msg = "Input volume shape is not a 3-tuple."
        raise ValueError(msg)
    shape: tuple[int, int, int] = vol_shape
    block_dims: tuple[int, int, int] = tuple(map(min, shape, _BLOCKDIMS))  # ty:ignore[invalid-assignment]
    grid_dims: tuple[int, int, int] = tuple(
        map(lambda s, b: math.ceil(s / b), shape, block_dims)
    )
    return grid_dims, block_dims


def _get_c_type_name(arr: np.ndarray | cp.ndarray) -> str:
    """E.g. convert 'np.float32' to 'float' and 'np.uint8' to 'unsigned short char'."""
    return _c_type_name_dict[arr.dtype]


def sample_FORBILD(  # noqa: C901, N802
    shapes: list[fv.Shape] | np.ndarray[np.dtype[fv.Shape]],  # ty:ignore[invalid-type-arguments]
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    *,
    dtype: np.dtype | None = None,
    out: cp.ndarray | None = None,
    supersamp_cubic_vox_wid: float | None = None,
    supersamp_factor: int | None = None,
) -> cp.ndarray[
    tuple[int, int, int], np.dtype[np.float16 | np.float32 | np.float64]
]:
    """Convert to a cupy array by simple point-sampling."""
    # Handle lists and floats etc by converting to np.ndarray
    xs = np.array(xs)
    ys = np.array(ys)
    zs = np.array(zs)
    xs = np.reshape(xs, (xs.size, 1, 1)).astype(np.float32)
    ys = np.reshape(ys, (1, ys.size, 1)).astype(np.float32)
    zs = np.reshape(zs, (1, 1, zs.size)).astype(np.float32)
    shape = (xs.size, ys.size, zs.size)

    # Validate supersamp args
    if (supersamp_cubic_vox_wid is None) != (supersamp_factor is None):
        msg = "Provide either one or both of 'supersamp_cubic_vox_wid' and 'supersamp_factor'."
        raise ValueError(msg)
    if supersamp_cubic_vox_wid is None:
        supersamp_cubic_vox_wid = 0
    if supersamp_factor is None:
        supersamp_factor = 1

    # Validate out
    if out is None:
        out = cp.zeros(
            shape, dtype=np.float32, order="C"
        )  # explicit ordering to C may help with the CUDA kernel coalescing memory writes
    else:
        if type(out) is np.ndarray:
            msg = "Must provide a cp.ndarray to this method, not a np.ndarray."
            raise ValueError(msg)
        if out.shape != shape:
            msg = "Mismatch in out.shape and the shape implied by the xs, ys, zs."
            raise ValueError(msg)
        out.fill(0)  # Fill with zeros, since the kernel will add to it

    # Validate dtype and out.dtype
    if dtype is None:
        if out is None:
            dtype = np.float32  # ty:ignore[invalid-assignment]
        else:
            dtype = out.dtype
    else:
        if out is not None:
            if dtype != out.dtype:
                msg = "Mismatch in the dtype specified and the dtype of the output volume."
                raise ValueError(msg)
    if dtype not in (np.float16, np.float32, np.float64):
        msg = f"Unsupported dtype: {dtype!s}"
        raise ValueError(msg)
    if out is not None:
        if out.dtype not in (np.float16, np.float32, np.float64):
            msg = f"Unsupported dtype: {out.dtype!s}"
            raise ValueError(msg)

    cp_shapes_arr = cp.array(shapes, dtype=fv.Shape)
    ctype = _get_c_type_name(out)
    kern = _module.get_function(
        f"add_FORBILD_sampling<{ctype:s},FORBILD_Shape>"
    )
    args = (
        _tuple_to_uint3(shape),
        cp.array(xs, dtype=cp.float32),
        cp.array(ys, dtype=cp.float32),
        cp.array(zs, dtype=cp.float32),
        cp_shapes_arr,
        np.uint32(cp_shapes_arr.size),
        _tuple_to_ulong3(out.strides),
        out,
        np.float32(supersamp_cubic_vox_wid),
        np.uint8(supersamp_factor),
    )
    kern(*_get_grid_block_dims(shape), args)
    return out
