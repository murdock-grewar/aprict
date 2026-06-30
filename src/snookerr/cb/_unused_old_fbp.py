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

# fbp.py

"""Methods to compute filtered backprojecitons for the cylindrical CBCT geometry."""

import functools
import itertools
import math

import cupy as cp
import numpy as np

from ..cuda import header_injection, structs
from ..cuda.kern_structs import _np_dtype_to_c_type


@functools.lru_cache(maxsize=10)
def _get_kernel(dtype: np.dtype, axis_len: int) -> cp.RawKernel:
    dtype_str = _np_dtype_to_c_type[dtype]
    module: cp.RawModule = header_injection.load_injected_module(
        "./ramlak_convolve.cu",
        name_expressions=[
            f"convolve_with_ramlak_x<{t1:s},{t2:d}>"
            for (t1, t2) in itertools.product(
                [dtype_str],
                [axis_len],
            )
        ],
    )
    # module.compile()
    function: cp.RawKernel = module.get_function(
        "convolve_with_ramlak_x<{t1:s},{t2:d}>".format(
            t1=dtype_str, t2=axis_len
        )
    )
    return function


def ramp_filter_1d_smallest_stride_axis(
    cparr: cp.ndarray, physpixwid: float = 1
) -> cp.ndarray:
    """Apply a 1D ramp filter to the array, along the axis of lowest stride in memory."""
    axes = [0, 1, 2]
    axis: int = min(axes, key=lambda a: cparr.strides[a])
    del axes[axis]  # leave the remaining two axes in the list
    axes_ordered = [axis, axes[0], axes[1]]

    # Figure out how many blocks of max size 1024 are required
    numxblocks = (cparr.shape[axis] + (1024 - 1)) // 1024
    xblocksize = math.ceil(
        cparr.shape[axis] / numxblocks
    )  # make blocks evenly sized
    block_dims = (xblocksize, 1, 1)
    grid_dims = (
        numxblocks,
        cparr.shape[axes_ordered[1]],
        cparr.shape[axes_ordered[2]],
    )
    bytes_per_block = cparr.itemsize * math.prod(block_dims)
    if bytes_per_block > 48 * 1024:
        msg = "Convolution axis too long for this method to handle! This is based on a shared memory size of 48kB."
        raise NotImplementedError(msg)
    args = (
        cparr,
        np.array(
            tuple(np.array(cparr.shape)[axes_ordered].tolist()),
            dtype=structs.uivec3,
        ),  # unsigned int
        np.array(
            tuple(np.array(cparr.strides)[axes_ordered].tolist()),
            dtype=structs.ulvec3,
        ),  # unsigned long
        np.float32(physpixwid),
    )
    kern = _get_kernel(cparr.dtype, cparr.shape[axis])
    kern(grid_dims, block_dims, args)


if __name__ == "__main__":
    ncycles: int = 100
    cparr = cp.zeros((10, 4000, 3001), dtype=cp.float32).transpose((0, 2, 1))
    cparr[:, cparr.shape[1] // 2, :] = 1
    # image_view.plot_projections(cparr)
    import time

    from .. import image_view

    t1 = time.time()
    for _ in range(ncycles):
        cparr.fill(0)
        cparr[:, cparr.shape[1] // 2, :] = 1
        ramp_filter_1d_smallest_stride_axis(cparr, physpixwid=1)
    cp.cuda.get_current_stream().synchronize()
    t2 = time.time()
    print(
        f"Milliseconds elapsed per trial per projection: {1000 * (t2 - t1) / ncycles / cparr.shape[0]:.0f}."
    )

    image_view.plot_projections(cparr)
