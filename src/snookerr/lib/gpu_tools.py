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

"""Tools to probe GPU status and set some related CuPy settings."""

import time
import typing

import cupy as cp
import gpustat

# IMPORTANT:
# Further down, we run 'cupy_disable_pinned_mem_pool()' as part of the import of this module. This ensures that anything importing this module will not allocate any more pinned memory than it has to! An alternative approach would be to abstract away the cupyx.pinned_...() methods into methods in this module...


def clear_cache(gpu_index: int | None = None) -> None:
    """Clear cupy's cuda memory pool."""
    # ^ It doesn't do this automatically; at least not reliably. Even when Python objects go out of scope."""
    gpu_index: int = (
        cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    )
    with cp.cuda.Device(gpu_index):
        cp.get_default_memory_pool().free_all_blocks()


def utilisation_average_perc(gpu_indices: typing.Iterable[int]) -> float:
    """Return average percentage utilisation of selected GPUs."""
    # Get a set of utilisation percentages
    percs = {
        d["utilization.gpu"]
        for d in gpustat.new_query().jsonify()["gpus"]
        if d["index"] in gpu_indices
    }
    return sum(percs) / len(percs)


class UtilisationAveragePercMon:
    """Class to track GPU usage percentage over time."""

    gpu_indices: list[int]
    last_time: float | None
    interval_seconds: float
    cached_perc: float | None

    def __init__(
        self, gpu_indices: typing.Iterable[int], interval_seconds: float
    ) -> None:
        self.gpu_indices = list(gpu_indices)
        self.last_time = None
        self.cached_perc = None
        self.interval_seconds = interval_seconds

    @property
    def gpu_perc(self) -> float | None:
        """Return average percentage utilisation of the configured GPUs."""
        tim = float(time.time())
        if self.last_time is None or (
            self.last_time <= (tim - self.interval_seconds)
        ):
            self.last_time = tim
            self.cached_perc = utilisation_average_perc(self.gpu_indices)
        return self.cached_perc


def all_gpu_indices() -> set[int]:
    """Fetch a set of all available GPU indices."""
    gstat = gpustat.new_query().jsonify()
    return {gpu_dict["index"] for gpu_dict in gstat["gpus"]}


def clear_pinned_cache(gpu_index: int | None = None) -> None:
    """Clear cupy's pinned memory pool which resides on the host."""
    # ^It doesn't do this automatically. Even when Python objects go out of scope."""
    gpu_index: int = (
        cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    )
    with cp.cuda.Device(gpu_index):
        cp.get_default_pinned_memory_pool().free_all_blocks()


def get_gpu_mb_used_and_total(gpu_index: int | None = None) -> tuple[int, int]:
    """Return a tuple of (MB_used, MB_total) for a given GPU."""
    gpu_index: int = (
        cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    )
    gstat = gpustat.new_query().jsonify()
    gstat = [
        gpu_dict
        for gpu_dict in gstat["gpus"]
        if gpu_dict["index"] == gpu_index
    ]
    gstat = gstat[0]
    mb_used = gstat["memory.used"]
    mb_total = gstat["memory.total"]
    return mb_used, mb_total


def get_min_vram_mb_capacity_across_gpus(
    gpu_indices: list[int] | set[int],
) -> int:
    """Return the minimum number of MB of memory capacity across the given GPUs."""
    if len(gpu_indices) == 0:
        return 0
    gpu_indices = list(gpu_indices)
    mb_total = get_gpu_mb_used_and_total(gpu_indices[0])[1]
    for gpu in gpu_indices[1:]:
        mb_total = min(mb_total, get_gpu_mb_used_and_total(gpu)[1])
    return mb_total


def vram_usage_string(gpu_index: int | None = None) -> str:
    """Return a human-readable string describing the VRAM usage on the specified GPU."""
    gpu_index: int = (
        cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    )
    mb_used, mb_total = get_gpu_mb_used_and_total(gpu_index)
    return f"VRAM usage on GPU {gpu_index:02d}: {mb_used:d} / {mb_total} MiB"


def cupy_set_fft_plan_cache_size(n: int, gpu_index: int | None = None) -> None:
    """Set the cupy fft plan cache size on the given GPU."""
    gpu_index: int = (
        cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    )
    with cp.cuda.Device(gpu_index):
        cache = cp.fft.config.get_plan_cache()
        cache.set_size(n)


def cupy_cap_default_mem_pool(
    multiplier: int | float = 1,
    gpu_index: int | set[int] | list[int] | None = None,
) -> None:
    """Set the max allowed memory capacity cupy can use on the given GPU, as a multiple of GPU capacity."""
    gpu_index = cp.cuda.runtime.getDevice() if gpu_index is None else gpu_index
    if type(gpu_index) in (set, list):
        gpu_indices = set(gpu_index)  # ty:ignore[invalid-argument-type]
    elif type(gpu_index) is int:
        gpu_indices = {gpu_index}
    else:
        msg: str = (
            f"Unrecognised type return for GPU index: '{type(gpu_index)!s}'."
        )
        raise TypeError(msg)
    for gpu_index in gpu_indices:
        with cp.cuda.Device(gpu_index):
            _, tot = get_gpu_mb_used_and_total()
            cp.get_default_memory_pool().set_limit(
                size=int(tot * 1024**2 * multiplier)
            )  # Makes sure CuPy won't use more than the available VRAM. However, if allocating memory not managed by CuPy, such as creating CUDAArray objects, then the limit may be exceeded anyway.


# Doesn't work!
# def cupy_reset_pinned_mem_pool() -> None:
#     "Force-flushes the pinned memory pool by resetting it to the default. Gets around a CuPy bug where the pinned memory refuses to flush."
#     cp.cuda.set_pinned_memory_allocator(None)
#     clear_pinned_cache()
#     cp.cuda.set_pinned_memory_allocator(cp.get_default_pinned_memory_pool())


def cupy_disable_pinned_mem_pool() -> None:
    """Disabled cupy's pinned memory pool."""
    cp.cuda.set_pinned_memory_allocator(None)
