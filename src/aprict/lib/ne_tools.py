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

"""Tools related to numexpr usage."""

import enum
import itertools
import logging  # ruff: ignore[typing-only-standard-library-import]
import math
import os
import pathlib  # ruff: ignore[typing-only-standard-library-import]

import numexpr as ne
import numpy as np

from .. import logtools


def fastcopy(
    *,
    dst: np.ndarray | np.memmap,
    src: np.ndarray | np.memmap | int | float | bool,
) -> None:
    (
        """Copy data from `src` to `dst`.

    Parameters
    ----------
    dst : np.ndarray | np.memmap
        The destination numpy array view. Can be a view into a memory-mapped array.
    src : np.ndarray | np.memmap | int | float | bool
        The source of the data. Can be a numpy array view of matching shape, or a constant.

    """
        ""
    )
    ne.evaluate("src", out=dst, casting="same_kind", local_dict={"src": src})


def set_array_to_constant(
    arr: np.ndarray | np.memmap, *, const: int | float
) -> None:
    """Fill an array with a constant value."""
    const_arr = const * np.ones(((1,) * arr.ndim), dtype=arr.dtype)
    ne.evaluate("const_arr", out=arr, local_dict={"const_arr": const_arr})


def set_array_to_constant_pbar(
    arr: np.ndarray | np.memmap,
    *,
    const: int | float,
    pbar_desc: str,
    pbar_valunit: str | None,
    pbar_logger: logging.Logger | None,
    pbar_persist: bool,
    GiB_per_step: float = 1,  # ruff: ignore[invalid-argument-name]
) -> None:
    """Fill an array with a constant value, with a progress bar."""
    const_arr = const * np.ones(((1,) * arr.ndim), dtype=arr.dtype)

    # Depending on whether the array is F-ordered or C-ordered, slice it differently
    step_last_axis: bool = arr.strides[-1] >= arr.strides[0]
    if step_last_axis:
        bytes_per_layer = math.prod(arr.shape[:-1]) * arr.itemsize
        numlayers: int = arr.shape[-1]
    else:
        bytes_per_layer = math.prod(arr.shape[1:]) * arr.itemsize
        numlayers: int = arr.shape[0]

    layers_per_step = max(
        1, math.floor(GiB_per_step * 1024**3 / bytes_per_layer)
    )
    sl = slice(0, layers_per_step)
    with logtools.ProgressBar(
        desc=pbar_desc,
        maxval=numlayers * bytes_per_layer,
        valunit=pbar_valunit,
        unit_is_bytes=True,
        logger=pbar_logger,
        persist=pbar_persist,
    ) as pbar:
        while sl.start <= numlayers:
            arr_view = arr[..., sl] if step_last_axis else arr[sl, ...]
            ne.evaluate(
                "const_arr", out=arr_view, local_dict={"const_arr": const_arr}
            )
            pbar.update_to(sl.stop * bytes_per_layer)
            sl = slice(
                sl.start + layers_per_step,
                min(sl.stop + layers_per_step, numlayers),
            )


def downbin(  # ruff: ignore[complex-structure]
    x: np.ndarray,
    *,
    n: int,
    preserve_sum: bool,
    out: np.ndarray | None = None,
    axis: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Downbin or downsample an array to a common integer divisor in every axis."""
    # Normalize the 'axis' input value
    if axis is None:
        axis = tuple(range(x.ndim))
    elif type(axis) is int:
        axis = (axis,)
    # Check that a valid selection of axes was made
    for i in axis:
        if i >= x.ndim:
            msg = f"axis index {i:d} too large for array with dimensions {x.ndim}"
            raise ValueError(msg)
    if len(axis) == 0:
        return (
            x.copy()
        )  # no downsampling; just return a copy of the original array

    if n == 1 and (out is None or out is x):
        return x
    newshape = tuple((s if s in axis else s // n) for s in x.shape)
    xnew = np.empty_like(x, shape=newshape) if out is None else out
    first: bool = True

    for offsets in itertools.product(range(n), repeat=len(axis)):
        # Create slice objects
        sls = []
        counter: int = 0
        for i in range(x.ndim):
            if i in axis:
                sls.append(slice(offsets[counter], None, n))
                counter += 1
            else:
                sls.append(slice(None, None))

        xsl = x[*sls]
        if first:
            ne.evaluate("xsl", out=xnew, local_dict={"xsl": xsl, "xnew": xnew})
            first = False
        else:
            ne.evaluate(
                "xsl + xnew", out=xnew, local_dict={"xsl": xsl, "xnew": xnew}
            )
    if not preserve_sum:
        ne.evaluate(
            "xnew/num",
            out=xnew,
            local_dict={"xnew": xnew, "num": pow(n, len(axis))},
        )
    return xnew


def save(path: pathlib.Path, arr: np.ndarray) -> None:
    """Save an array to a path in a faster manner than np.save."""
    mm: np.memmap = np.lib.format.open_memmap(
        path, shape=arr.shape, dtype=arr.dtype, mode="w+"
    )
    ne.evaluate("x", out=mm, local_dict={"x": arr})


class NETHREADS(enum.Enum):
    """Presets for how to control the number of numexpr threads."""

    NE_DEFAULT = 1  # default numexpr behaviour--- don't change anything
    NE_DEFAULT_MAXTHREADS = 2  # the maximum number of threads which numexpr allow with current environment varibales.
    ALL_LOGICAL_CORES = 3  # launch as many threads as there are logical cores available to this process. This will throw an error if unable to set the maximum.


def try_set_nethreads(
    nethreads: NETHREADS, logger: logging.Logger | None = None
) -> int:
    """Import the numexpr module with the specified thread configuration."""
    # Set the target number of threads
    env_num_threads = os.environ.get("NUMEXPR_NUM_THREADS", None)
    logger and logger.debug(
        f"Value of env 'NUMEXPR_NUM_THREADS': {env_num_threads!s:s}"
    )
    env_num_threads: int | None = (
        None if env_num_threads is None else int(env_num_threads)
    )
    env_max_threads = os.environ.get("NUMEXPR_MAX_THREADS", None)
    logger and logger.debug(
        f"Value of env 'NUMEXPR_MAX_THREADS': {env_max_threads!s:s}"
    )
    env_max_threads: int | None = (
        None if env_max_threads is None else int(env_max_threads)
    )
    logger and logger.info(f"Setting for --ne-threads: {nethreads.name:s}")
    match nethreads:
        case NETHREADS.NE_DEFAULT:
            nthreads_target: int | None = (
                None  # env_num_threads or env_max_threads
            )
        case NETHREADS.NE_DEFAULT_MAXTHREADS:
            nthreads_target: int = ne.MAX_THREADS
        case NETHREADS.ALL_LOGICAL_CORES:
            nthreads_target: int = int(
                os.process_cpu_count()
            )  # number of logical cores
            if env_max_threads is None or env_max_threads < nthreads_target:
                # Set environment variable to enable a larger number of threads
                os.environ["NUMEXPR_MAX_THREADS"] = str(nthreads_target)
            # import numexpr now that the env NUMEXPR_MAX_THREADS has been set appropriately.
        case _:
            msg = f"Unrecognized choice for nethreads: '{nethreads.name}'"
            raise NotImplementedError(msg)

    # Try to set the target number of threads:
    if nthreads_target is not None:
        logger and logger.debug(
            f"Target number of numexpr threads: {nthreads_target:d}"
        )
        ne.set_num_threads(nthreads_target)
        nthreads_actual = ne.get_num_threads()
        if nthreads_actual != nthreads_target:
            logger and logger.warning(
                "Failed to set the target number of numexpr threads."
            )
    nthreads_actual = ne.get_num_threads()
    logger and logger.info(f"Number of numexpr threads: {nthreads_actual:d}")

    return nthreads_actual
