# Copyright (C) 2026  Murdock Grewar
#
# This:program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, see <https://www.gnu.org/licenses/>.

# volume.py
"""Class to abstract away the details of a 3D voxelised volume."""
# Developer's note: this class is intended to contain the end-user API
# to the volume data. The user should not have to interact with internal
# classes like texture_tools.Texture, and the like.

import numpy as np

from .lib import gpu_tools, texture_tools
from .lib.texture_tools import operations as tops

_ShapeType = tuple[int, int, int]
_DType = np.dtype[np.float16] | np.dtype[np.float32] | np.dtype[np.float64]


class Volume[
    Shape: _ShapeType,  # Shape will be a subtype, like tuple[Literal[10], Literal[10], Literal[10]]
    DType: _DType,
]:
    """A rectangular prism of voxels. Underpinned by a CUDA surface object."""

    # Used to store the TypeVars which are inferred from the input.
    _shape: _ShapeType
    _dtype: _DType

    # Other fields
    _texture: texture_tools.Texture

    @property
    def shape(self) -> _ShapeType:
        """Return the shape of the volume array."""
        return self._shape

    @property
    def dtype(self) -> _DType:
        """Return the dtype of the volume array."""
        return self._dtype

    def __init__(
        self,
        arr: np.ndarray[
            Shape, DType
        ],  # this typing ensures the correct dimensions (3) and a supported numpy dtype
    ) -> None:
        # Essentially, we are storing the resolved types of the TypeVars
        self._dtype = arr.dtype
        self._shape = arr.shape
        # Create the texture and load the data
        self._texture = self._new_texture(shape=self._shape, dtype=self._dtype)
        self.copy_from(arr)

    def __del__(self) -> None:
        """Object cleanup."""
        gpu_tools.clear_cache()  # Ensure that CuPy cleans up the used memory.

    @classmethod
    def _new_texture(cls, shape: Shape, dtype: DType) -> texture_tools.Texture:
        return texture_tools.Texture(
            (dtype, shape),
            address_mode=texture_tools.AddressMode.cudaAddressModeBorder,
            filter_mode=texture_tools.FilterMode.cudaFilterModePoint,
            layered=False,
            normalise_coordinates=False,
            include_texture=True,
            include_surface=True,
        )

    @classmethod
    def zeros(
        cls, shape: _ShapeType, dtype: _DType = np.dtype(np.float32)
    ) -> Volume:
        """Initialise a new volume, filled with zeros."""
        if type(shape) is not tuple or len(shape) != 3:
            msg = "Invalid shape provided. It should be a tuple of three positive integers."
            raise ValueError(msg)
        self: Volume = cls.__new__(cls)
        self._dtype = dtype
        self._shape = shape
        self._texture = self._new_texture(shape=self._shape, dtype=self._dtype)
        self.fill(0)
        return self

    def fill(self, value: int | float) -> None:
        """Fill the volume with a constant value."""
        tops.fill(self._texture, value)

    def copy_from(self, arr: np.ndarray[Shape, DType]) -> None:
        """Copy a numpy/cupy array into this Volume object."""
        self._texture.cuda_array.copy_from(arr)

    def copy_to(self, arr: np.ndarray[Shape, DType]) -> None:
        """Copy this Volume into a numpy/cupy array."""
        self._texture.cuda_array.copy_to(arr)
