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
import logging  # noqa: TC003
import os
import pathlib  # noqa: TC003

import numexpr as ne
import numpy as np


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
            nthreads_target: int | None = env_num_threads or env_max_threads
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
