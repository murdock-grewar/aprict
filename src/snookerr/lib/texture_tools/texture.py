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

"""Texture type to encapsulate CUDA textures and surfaces."""

import enum
import itertools
import math

import cupy as cp
import numpy as np

__all__ = [
    "AddressMode",
    "FilterMode",
    "Texture",
]

# CUDA doesn't seem to support 64-bit floats in channel descriptors. See https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__HIGHLEVEL.html#group__CUDART__HIGHLEVEL_1g655725c27d8ffe75accb9b531ecf2d15 where it says "The template function specializes for the following scalar types: char, signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long, and float." Note that double isn't listed. When I try a double, I get 'cudaErrorInvalidChannelDescriptor'.
# Despite the documentation saying that long and unsigned long should be supported, that doesn't seem to be the case, as I get the same error. I'm not sure why that is.
# TODO: find out why 64-bit values do not work.  # ruff: ignore[line-contains-todo]

_supported_arr_dtypes: set[cp.dtype] = {
    cp.dtype(cp.int8),
    cp.dtype(cp.int16),
    cp.dtype(cp.int32),
    # cp.dtype(cp.int64),
    cp.dtype(cp.uint8),
    cp.dtype(cp.uint16),
    cp.dtype(cp.uint32),
    # cp.dtype(cp.uint64),
    cp.dtype(cp.float16),
    cp.dtype(cp.float32),
    # cp.dtype(cp.float64),
}

# The CUDA channel format kinds corresponding to the different dtypes.
# For more info on channel format kinds, see the CUDA documentation at:
# https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html#group__CUDART__TYPES_1g8085eac5cb54b4228f3619a60f235119
_dtype_to_cfk: dict[cp.dtype, int] = {
    cp.dtype(cp.int8): cp.cuda.runtime.cudaChannelFormatKindSigned,
    cp.dtype(cp.int16): cp.cuda.runtime.cudaChannelFormatKindSigned,
    cp.dtype(cp.int32): cp.cuda.runtime.cudaChannelFormatKindSigned,
    # cp.dtype(cp.int64): cp.cuda.runtime.cudaChannelFormatKindSigned,
    cp.dtype(cp.uint8): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
    cp.dtype(cp.uint16): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
    cp.dtype(cp.uint32): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
    # cp.dtype(cp.uint64): cp.cuda.runtime.cudaChannelFormatKindUnsigned,
    cp.dtype(cp.float16): cp.cuda.runtime.cudaChannelFormatKindFloat,
    cp.dtype(cp.float32): cp.cuda.runtime.cudaChannelFormatKindFloat,
    # cp.dtype(cp.float64): cp.cuda.runtime.cudaChannelFormatKindFloat,
}


# ruff: disable[mixed-case-variable-in-class-scope]
class FilterMode(enum.Enum):
    """CUDA filter modes for textures."""

    cudaFilterModePoint = cp.cuda.runtime.cudaFilterModePoint
    cudaFilterModeLinear = cp.cuda.runtime.cudaFilterModeLinear


class AddressMode(enum.Enum):
    """CUDA address modes for textures."""

    cudaAddressModeWrap = cp.cuda.runtime.cudaAddressModeWrap  # = 0
    cudaAddressModeClamp = cp.cuda.runtime.cudaAddressModeClamp  # = 1
    cudaAddressModeMirror = cp.cuda.runtime.cudaAddressModeMirror  # = 2
    cudaAddressModeBorder = cp.cuda.runtime.cudaAddressModeBorder  # = 3


# ruff: enable[mixed-case-variable-in-class-scope]


class Texture:
    """An encapsulating class for a CUDA texture and/or surface."""

    texobj: cp.cuda.texture.TextureObject
    surfobj: cp.cuda.texture.SurfaceObject
    tdesc: cp.cuda.texture.TextureDescriptor  # Equivalent to cudaTextureDesc
    rdesc: cp.cuda.texture.ResourceDescriptor  # Equivalent to cudaResourceDesc
    cuda_array: cp.cuda.texture.CUDAarray
    shape: tuple[int, int] | tuple[int, int, int]
    nbytes: int
    dtype: np.dtype

    @property
    def is_flagset_layered(self) -> bool:
        """The state of the 'cudaArrayLayered' flag."""
        # For some reason, the definition "cudaArrayLayered = 1" has been commented out in the CuPy source code.
        return self.cuda_array.flags & 1

    @property
    def is_flagset_surfaceLoadStore(self) -> bool:  # ruff: ignore[invalid-function-name]
        """The state of the 'cudaArraySurfaceLoadStore' flag."""
        return (
            self.cuda_array.flags & cp.cuda.runtime.cudaArraySurfaceLoadStore
        )

    @property
    def is_texture_included(self) -> bool:
        """Whether this object has a corresponding cudaTexture."""
        return self.texobj is not None

    @property
    def is_surface_included(self) -> bool:
        """Whether this object has a corresponding cudaSurface."""
        return self.surfobj is not None

    @property
    def filter_mode(self) -> int:
        """The filterMode of the texture descriptor."""
        return self.texobj.TexDesc.get_texture_desc()["filterMode"]

    @property
    def address_modes(self) -> int:
        """The addressModes of the texture descriptors."""
        return self.texobj.TexDesc.get_texture_desc()["addressModes"]

    # For simplicity, I am currently making this always generate both a texture and a surface
    # I'm going to strip away features/options I don't use, for the time being, like normalised coordinates.
    def __init__(  # ruff: ignore[complex-structure]
        self,
        arraylike_or_dtypeshapetuple: np.ndarray
        | np.memmap
        | cp.ndarray
        | tuple[cp.dtype, tuple[int, int]]
        | tuple[
            cp.dtype, tuple[int, int, int]
        ],  # | cp.cuda.texture.CUDAarray,
        *,
        # Named arguments
        address_mode: AddressMode
        | tuple[AddressMode, AddressMode]
        | tuple[AddressMode, AddressMode, AddressMode],
        filter_mode: FilterMode,
        # Options with defaults
        layered: bool = False,  # This option can only be set to True when the array is 3D
        normalise_coordinates: bool = False,
        include_texture: bool = True,
        include_surface: bool = True,
    ) -> None:

        # Validate that a proper address_mode has been passed
        # Note from NVIDIA's C++ programming guide:
        # "cudaAddressModeWrap and cudaAddressModeMirror are only supported for normalized texture coordinates"
        # Unfortunately, the failure is silent if you get it wrong. Fetched values will be wrong, but no error is thrown.
        msg: str = "NVIDIA only supports cudaAddressModeWrap and cudaAddressModeMirror when using normalised texture coordinates."
        if type(address_mode) is AddressMode:
            if address_mode in (
                AddressMode.cudaAddressModeWrap,
                AddressMode.cudaAddressModeMirror,
            ):
                if not normalise_coordinates:
                    raise ValueError(msg)
        elif type(address_mode) is tuple:
            if (
                len(
                    set(address_mode).intersection(
                        {
                            AddressMode.cudaAddressModeWrap,
                            AddressMode.cudaAddressModeMirror,
                        }
                    )
                )
                > 0
            ):
                raise ValueError(msg)
        else:
            msg = "Unknown type for address_mode. Either pass an AddressMode, or a tuple of them."
            raise TypeError(msg)

        # Get the dtype and shape. These are needed immediately, for argument validation.
        dtype: cp.dtype
        shape: tuple[int, int] | tuple[int, int, int]
        nbytes: int
        if type(arraylike_or_dtypeshapetuple) is tuple:
            tup = arraylike_or_dtypeshapetuple
            dtype, shape = cp.dtype(tup[0]), tup[1]
        elif type(arraylike_or_dtypeshapetuple) in (
            np.ndarray,
            np.memmap,
            cp.ndarray,
        ):
            arr = arraylike_or_dtypeshapetuple
            dtype, shape = cp.dtype(arr.dtype), arr.shape  # ty:ignore[unresolved-attribute]
        elif type(arraylike_or_dtypeshapetuple) is cp.cuda.texture.CUDAarray:
            raise NotImplementedError()
            shape = (
                arraylike_or_dtypeshapetuple.depth,
                arraylike_or_dtypeshapetuple.height,
                arraylike_or_dtypeshapetuple.width,
            )
        else:
            msg = f"Unknown type for the arraylike_or_shape input: '{type(arraylike_or_dtypeshapetuple)!s}'."
            raise TypeError(msg)
        nbytes = dtype.itemsize * math.prod(shape)

        # Validate the shape
        if len(shape) != 2 and len(shape) != 3:
            msg = f"The input array or shape specifies an array that is not 2D or 3D. The input shape is '{shape!s}'."
            raise ValueError(msg)

        # Validate that the dtype is supported
        if dtype not in _supported_arr_dtypes:
            msg = f"The array dtype '{dtype!s}' is not (yet?) supported."
            raise TypeError(msg)

        # Validate that the address_mode provided matches the shape, if applicable
        if type(address_mode) is tuple:
            if len(address_mode) != len(shape):
                msg = "Mismatch in dimension of array of the number of address_mode supplied."
                raise ValueError(msg)

        # BEGIN creating the texture/surface
        # Channel format kind
        cfk: int = _dtype_to_cfk[dtype]
        # Channel format descriptor
        nbits_per_el = (
            dtype.itemsize * 8
        )  # dtype.itemsize gives bytes. We need bits.
        cfd = cp.cuda.texture.ChannelFormatDescriptor(
            nbits_per_el,  # x channel
            0,
            0,
            0,  # y, z, w channels
            f=cfk,
        )
        # Reorder the 'shape' tuple to be compatible with the CUDA interface
        cuda_shape: tuple[int, int, int] = (
            (shape[1], shape[0], 0)
            if len(shape) == 2
            else tuple(reversed(shape))
        )
        # ---- Flags ----
        # From the CUDA toolkit documentation: https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html
        #     The flags parameter enables different options to be specified that affect the allocation, as follows.
        #
        #     cudaArrayDefault: This flag's value is defined to be 0 and provides default array allocation
        #     cudaArrayLayered: Allocates a layered CUDA array, with the depth extent indicating the number of layers
        #     cudaArrayCubemap: Allocates a cubemap CUDA array. Width must be equal to height, and depth must be six. If the cudaArrayLayered flag is also set, depth must be a multiple of six.
        #     cudaArraySurfaceLoadStore: Allocates a CUDA array that could be read from or written to using a surface reference.
        #     cudaArrayTextureGather: This flag indicates that texture gather operations will be performed on the CUDA array. Texture gather can only be performed on 2D CUDA arrays.
        #     cudaArraySparse: Allocates a CUDA array without physical backing memory. The subregions within this sparse array can later be mapped onto a physical memory allocation by calling cuMemMapArrayAsync. This flag can only be used for creating 2D, 3D or 2D layered sparse CUDA arrays. The physical backing memory must be allocated via cuMemCreate.
        #     cudaArrayDeferredMapping: Allocates a CUDA array without physical backing memory. The entire array can later be mapped onto a physical memory allocation by calling cuMemMapArrayAsync. The physical backing memory must be allocated via cuMemCreate.
        #
        flags = cp.cuda.runtime.cudaArrayDefault
        if include_surface:
            flags = flags | cp.cuda.runtime.cudaArraySurfaceLoadStore
        if layered:
            # For some reason, the definition "cudaArrayLayered = 1" has been commented out in the CuPy source code.
            flags = flags | 1
        # ---------------
        cuda_arr: cp.cuda.texture.CUDAarray = cp.cuda.texture.CUDAarray(
            cfd,
            *cuda_shape,
            flags,
        )

        # Prepare the tuple of address modes. This also needs to be reversed, like cuda_shape
        cuda_address_modes: tuple[int, int] | tuple[int, int, int]
        if type(address_mode) is tuple:
            cuda_address_modes = tuple(
                e.value for e in tuple(reversed(address_mode))
            )  # Reverse it so that the user specifies them in the same order as the axes in the 'shape' input.
        elif type(address_mode) is AddressMode:
            cuda_address_modes = tuple(
                e.value
                for e in tuple(itertools.repeat(address_mode, len(shape)))
            )
        else:
            raise ValueError()
        # filterMode
        cuda_filter_mode: int = filter_mode.value
        # Initialise objects
        rdesc = None
        tdesc = None
        tobj = None
        # Resource descriptor
        if include_texture or include_surface:
            rdesc = cp.cuda.texture.ResourceDescriptor(
                restype=cp.cuda.runtime.cudaResourceTypeArray, cuArr=cuda_arr
            )
        # Create texture
        if include_texture:
            # Create texture object -- allows reading only. Is a Python wrapper for the CUDA C type cudaTextureObject_t
            tdesc = cp.cuda.texture.TextureDescriptor(
                addressModes=cuda_address_modes,
                filterMode=cuda_filter_mode,
                readMode=cp.cuda.runtime.cudaReadModeElementType,
                normalizedCoords=int(normalise_coordinates),
            )
            tobj = cp.cuda.texture.TextureObject(rdesc, tdesc)
        # Create surface
        if include_surface:
            # Create surface object -- allows reading AND writing. Is a Python wrapper for the CUDA C type cudaSurfaceObject_t
            sobj = cp.cuda.texture.SurfaceObject(rdesc)

        # Copy the contents of the CuPy array to the CUDAarray.
        if type(arraylike_or_dtypeshapetuple) in (
            np.ndarray,
            np.memmap,
            cp.ndarray,
        ):
            cuda_arr.copy_from(arraylike_or_dtypeshapetuple)
        elif type(arraylike_or_dtypeshapetuple) is tuple:
            pass
        else:
            raise NotImplementedError()

        # Assign self values
        self.texobj = tobj
        self.surfobj = sobj
        self.tdesc = tdesc
        self.rdesc = rdesc
        self.nbytes = nbytes
        self.cuda_array = cuda_arr
        self.shape = shape
        self.dtype = dtype
