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
]

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
