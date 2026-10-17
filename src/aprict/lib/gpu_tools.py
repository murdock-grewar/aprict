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

import queue
import random
import threading
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


def _gpu_perc_monitor_thread(
    q_in: queue.SimpleQueue,
    q_out: queue.SimpleQueue,
    gpu_indices: typing.Iterable[int],
    interval_seconds: float,
) -> None:
    if interval_seconds <= 0:
        msg = "interval_seconds must be a positive float"
        raise ValueError(msg)
    gpu_perc: float = utilisation_average_perc(gpu_indices)
    last_time = float(time.time())

    # Will loop, reading the queue. If True is received, then quit this thread.
    # If False is received, then send back the latest gpu_perc
    quit_: bool = False
    while not quit_:
        # Get the new timeout interval based on how long it's been since the last check.
        # Set a timeout with a small noise on the interval time.
        # This is to prevent spurious correlations with the samples being produced at exact regular intervals
        tim = float(time.time())
        interval_seconds_ = max(
            0,
            min(
                interval_seconds * 1.01,
                interval_seconds * (1 + 0.01 * (0.5 - random.random()))  # nosec
                - (tim - last_time),
            ),
        )
        if interval_seconds_ == 0:
            # Refresh right away, before checking the queue
            gpu_perc = utilisation_average_perc(gpu_indices)
            last_time = float(time.time())
            continue

        try:
            # Get the message
            quit_ = q_in.get(timeout=interval_seconds_)
            # A message was received, but it wasn't to quit.
            if not quit_:
                # Note: DO NOT recalculate the gpu usage here, because that would cause the GPU usage sampling times to correlate with when the GPU usage reading was requested.
                q_out.put((last_time, gpu_perc), block=False)
        except queue.Empty:
            # No message received. Fetch the latest usage and repeat.
            gpu_perc = utilisation_average_perc(gpu_indices)
            last_time = float(time.time())


class UtilisationAveragePercMon:
    """Class to track GPU usage percentage over time."""

    gpu_indices: list[int]
    interval_seconds: float
    cached_perc: float | None
    _monitor_thread: threading.Thread
    _monitor_thread_msg_queue_in: queue.SimpleQueue
    _monitor_thread_msg_queue_out: queue.SimpleQueue

    def __init__(
        self, gpu_indices: typing.Iterable[int], interval_seconds: float
    ) -> None:
        self.gpu_indices = list(gpu_indices)
        self.last_time = None
        self.cached_perc = None
        self.interval_seconds = interval_seconds
        self._monitor_thread_msg_queue_in = (
            queue.SimpleQueue()
        )  # SimpleQueue is FIFO, which is what we need
        self._monitor_thread_msg_queue_out = (
            queue.SimpleQueue()
        )  # SimpleQueue is FIFO, which is what we need
        self._monitor_thread = threading.Thread(
            target=_gpu_perc_monitor_thread,
            args=(
                self._monitor_thread_msg_queue_in,
                self._monitor_thread_msg_queue_out,
                tuple(self.gpu_indices),
                self.interval_seconds,
            ),
        )
        self._monitor_thread.start()

    @property
    def gpu_perc(self) -> float | None:
        """Average percentage utilisation of the configured GPUs."""
        # Send a message telling the object that we want it to send back the GPU usage
        self._monitor_thread_msg_queue_in.put(False)
        _measurement_time, gpu_perc = self._monitor_thread_msg_queue_out.get(
            block=True
        )
        return gpu_perc

    def close(self) -> None:
        """Gracefully close the background thread associated with this object."""
        if self._monitor_thread.is_alive():
            self._monitor_thread_msg_queue_in.put(True)
            self._monitor_thread.join()


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
