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
    "softmask_to_cylinder_xy",
]

import itertools
import math
from typing import TYPE_CHECKING

import numpy as np

from ...cuda import header_injection
from ...cuda.kern_structs import _np_dtype_to_c_type as _c_type_name_dict

if TYPE_CHECKING:
    import cupy as cp

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
