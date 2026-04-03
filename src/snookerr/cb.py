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

# cb.py

"""Interfaces to the cone-beam projection and backprojection CUDA code."""

import itertools
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import numpy.typing as npt  # noqa: F401

from .cuda import header_injection, structs
from .cuda.kern_structs import _np_dtype_to_c_type

if TYPE_CHECKING:
    from .volume import Volume


def injected_module_texture_chunked() -> cp.RawModule:  # noqa: D103
    # Inject and compile module
    module: cp.RawModule = header_injection.load_injected_module(
        "./CB_project_texture_chunked.cu",
        name_expressions=[
            f"project_CB_texture_chunked<{t1:s},{t2:s}>"
            for t1, t2 in itertools.product(
                ["half", "float"], ["float", "double"]
            )
        ],
    )
    module.compile()
    return module


_projection_module: cp.RawModule = injected_module_texture_chunked()


def _select_projection_kernel(  # noqa: ANN202
    config: np.dtype[  # noqa: ARG001
        structs.config_cb_project  # ty:ignore[invalid-type-arguments]
    ],  # object to pass in any miscellaneous configuration for the projection kernel
    volume: Volume,
    proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]],  # noqa: ARG001  # ty:ignore[invalid-type-arguments]
    output_array: cp.ndarray,
):
    t1 = _np_dtype_to_c_type[volume.dtype]
    t2 = _np_dtype_to_c_type[output_array.dtype]
    return _projection_module.get_function(
        f"project_CB_texture_chunked<{t1:s},{t2:s}>"
    )


def project_flatpanel(
    config: np.dtype[
        structs.config_cb_project  # ty:ignore[invalid-type-arguments]
    ],  # object to pass in any miscellaneous configuration for the projection kernel
    volume: Volume,
    proj_specs: np.ndarray[tuple[int, ...], np.dtype[structs.proj_spec]],  # ty:ignore[invalid-type-arguments]
    out: cp.ndarray[tuple[int, ...], cp.dtype[cp.float32 | cp.float64]]
    | None = None,
    block: bool = False,
) -> cp.ndarray:
    """Add the projection of the input volume to the output array."""
    output_array = out
    # Assert correct dtype for the input proj_specs
    if not proj_specs.dtype == structs.proj_spec.dtype:
        msg = "Invalid dtype for the input array. It should be an array of 'proj_spec' objects."
        raise ValueError(msg)
    # Assert that all have the type set to flat panel
    dspecs = np.array(proj_specs["dspec"], dtype=structs.det_spec.dtype)
    if not np.all(0 == dspecs["type"]):
        msg = "Only flat-panel detectors are currently supported. One or more of the provided proj_spec indicates a detector type other than flat panel."
        raise NotImplementedError(msg)
    # Assert that all detector dimensions are the same
    nus = dspecs["nu"].flatten()
    nvs = dspecs["nv"].flatten()
    if not np.all(nus == nus[0]) or not np.all(nvs == nvs[0]):
        msg = "The input detector dimensions are inconsistent between projections. This is not currently implemented."
        raise NotImplementedError(msg)
    # Check consistency with the output array shape
    det_shape = (nus[0], nvs[0])
    proj_array_shape = (
        proj_specs.shape
    )  # Note: the input could be any shape, not just a 1D input array
    if len(proj_array_shape) < 1:
        msg = "The input proj_specs array is not at least a 1D array. It should be at least a 1D array."
        raise ValueError(msg)
    data_shape = proj_array_shape + det_shape
    if output_array is None:
        # Create the output array
        output_array = cp.zeros(data_shape, dtype=np.float32)
    else:
        if output_array.shape != data_shape:
            msg = f"Invalid shape for the output_array. The expected shape is {data_shape}, but the provided output_array has shape {output_array.shape}."
            raise NotImplementedError(msg)
    # Prepare the kernel and the static kernel args
    blockx = 10
    blocky = 10
    blocklayer = 10
    kern = _select_projection_kernel(config, volume, proj_specs, output_array)
    block_dims = (blockx, blocky, blocklayer)
    grid_dims = (
        int(np.ceil(output_array.shape[-2] / blockx)),
        int(np.ceil(output_array.shape[-1] / blocky)),
        int(np.ceil(output_array.shape[-3] / blocklayer)),
    )  # gridDims:  (x,y,z)
    texchunkspec = np.array(
        (
            np.array((0, 0, 0), dtype=structs.uivec3.dtype),  # offsets
            np.array(volume.shape, dtype=structs.uivec3.dtype),  # shape
            np.array(
                tuple(-np.array(volume.shape, dtype=np.float32) / 2),
                dtype=structs.fvec3.dtype,
            ),  # texpos
        ),
        dtype=structs.vol_tex_chunk_spec.dtype,
    )
    input_proj_array_len = np.uint32(output_array.shape[-3])
    input_proj_array_strides = np.array(
        output_array.strides[-3:], dtype=structs.uivec3.dtype
    )
    # Compute the projection. Loop through higher dimensions of the passed array, if applicable.
    for index_tuple in itertools.product(*map(range, proj_array_shape[:-1])):
        # If a flat proj_array was passed, then 'index_tuple' will only take on the value of an  empty tuple.
        input_proj_specs_slice = cp.array(proj_specs[index_tuple])
        input_proj_array_ptr = np.uint64(
            output_array.data.ptr
            + sum(
                map(
                    lambda index, stride: index * stride,
                    index_tuple,
                    output_array.strides[:-3],
                ),
                0,
            )
        )
        # Launch the kernel
        args = (  # input args
            config,
            input_proj_specs_slice,
            input_proj_array_len,
            texchunkspec,
            volume._texture.surfobj,  # noqa: SLF001
            input_proj_array_strides,  # strides in bytes
            input_proj_array_ptr,  # pointer to the beginning of the det data
        )
        kern(grid_dims, block_dims, args)
    if block:
        cp.cuda.get_current_stream().synchronize()
    return output_array
