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

"""Tools to apply ramp filters."""

import enum
import itertools
import time
import typing

import cupy as cp
import cupyx.scipy.fft as cufft
import cupyx.scipy.ndimage
import numpy as np

from .. import ramp_kernels
from ..lib import gpu_tools


class RampFilterer:
    """A class to facilitate repetitive 1D filtrations along axis 1 of a cupy projection array."""

    # Note: tested two approaches:
    # 1. Applying a direct convolution formula by computing the components of the Ram-Lak filter
    # 2. Applying an FFT convolution.
    # To my surprise, approach 2 was faster, despite approach 1 being well-optimised with CUDA thread blocks and shared memory etc.
    # Therefore, I've fallen back on approach 2. I am assuming that the extra cost induced by padding the 2D projections etc will be worthwhile.

    chunk_shape: tuple[int, int, int]
    physpixwid: float
    stream: cp.cuda.Stream

    _cufft_plan_forward: cp.cuda.cufft.Plan1D
    _cufft_plan_bckward: cp.cuda.cufft.Plan1D
    _fft_mult_horz: cp.ndarray
    _chunk_mem_horz_filter: cp.ndarray
    _chunk_mem_horz_filter_after_fft: cp.ndarray

    # CUDA stream events
    _e_mem1_free: cp.cuda.Event | None
    _e_mem2_free: cp.cuda.Event | None

    def __init__(
        self,
        *,
        physpixwid: float = 1,
        proj_2d_shape: tuple[int, int],
        projs_per_batch: int = 2,  # not much benefit going above 3, I think. And it uses a lot of VRAM.
        stream: cp.cuda.Stream | None = None,
    ) -> None:
        self.chunk_shape = (projs_per_batch, *proj_2d_shape)
        self.physpixwid = physpixwid
        self.stream = (
            cp.cuda.get_current_stream() if stream is None else stream
        )
        self._initialise()

    def _initialise(self) -> None:
        with self.stream:
            chunk_shape = self.chunk_shape
            physpixwid = self.physpixwid
            # Get the horz kernel as a multiplier in Fourier space
            # kern_horz: cp.ndarray = cp.array(ramp_kernels.ramp_kernel(2*chunk_shape[1] - 1, physpixsize=physpixwid))
            kern_horz: cp.ndarray = cp.array(
                ramp_kernels.ramp_kernel_ramlak(
                    2 * chunk_shape[1] - 1, physpixsize=physpixwid
                )
            )
            # kern_horz: cp.ndarray = cp.array(ramp_kernels.ramp_kernel_softened(2*chunk_shape[1] - 1, inv_soften_strength=0.9, physpixsize=physpixwid))
            kern_horz = cp.pad(kern_horz, ((0, 1),), mode="constant")
            kern_horz = cp.roll(
                kern_horz, shift=1 - (kern_horz.shape[0] // 2), axis=0
            )
            self._fft_mult_horz = cufft.rfft(kern_horz, norm="backward")

            # Reserve memory and plans for FFT stuff
            self._chunk_mem_horz_filter = cp.empty(
                (chunk_shape[0], chunk_shape[1] * 2, chunk_shape[2]),
                dtype=cp.float32,
            )
            self._cufft_plan_forward = cufft.get_fft_plan(
                self._chunk_mem_horz_filter, axes=(1,), value_type="R2C"
            )
            self._chunk_mem_horz_filter_after_fft: cp.ndarray = cp.zeros(
                (
                    self._chunk_mem_horz_filter.shape[0],
                    self._chunk_mem_horz_filter.shape[1] // 2 + 1,
                    self._chunk_mem_horz_filter.shape[2],
                ),
                dtype=cp.complex64,
            )  # Initialise the output array
            self._cufft_plan_bckward = cufft.get_fft_plan(
                self._chunk_mem_horz_filter_after_fft,
                shape=(self._chunk_mem_horz_filter.shape[1],),
                axes=(1,),
                value_type="C2R",
            )

            self._e_mem1_free = None
            self._e_mem2_free = None

    def _filter_chunk_axis1(
        self, cparr_view: cp.ndarray, input_in_mem1: bool = False
    ) -> cp.ndarray:
        """Filter an input cp.ndarray (can be a view) and produce a view into a newly allocated cupy array as output."""
        with self.stream as stream:
            chunk_shape = self.chunk_shape
            # Check that the input shape fits with the internal chunk shape
            if not (cparr_view.shape[1], cparr_view.shape[2]) == (
                self.chunk_shape[1],
                self.chunk_shape[2],
            ):
                msg = "Input projection array shape does not match the shape initialised for this filtering object."
                raise ValueError(msg)
            if not cparr_view.shape[0] <= self.chunk_shape[0]:
                msg = "Tried to input too many projections at once (zeroth axis of input has a length longer than the chunk size of this filtering object)."
                raise ValueError(msg)
            # TODO check if this can be made more efficient by not filtering the unused projections at hte end of a chunk.  # noqa: FIX002

            # Do the horizontal filtration
            # An optimisation: if the input array is the chunk memory, then we don't need to copy it in.
            if input_in_mem1:
                self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], chunk_shape[1] :, :
                ] = 0
            else:
                if self._e_mem1_free is not None:
                    self._e_mem1_free.synchronize()
                self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], chunk_shape[1] :, :
                ] = 0
                self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], : chunk_shape[1], :
                ] = cparr_view[...]
            out = cufft.rfft(
                self._chunk_mem_horz_filter,
                axis=1,
                norm="backward",
                overwrite_x=True,
                plan=self._cufft_plan_forward,
            )
            self._e_mem1_free = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(self._e_mem1_free)

            if self._e_mem2_free is not None:
                self._e_mem2_free.synchronize()
            cp.multiply(
                out,
                self._fft_mult_horz[cp.newaxis, :, cp.newaxis],
                out=self._chunk_mem_horz_filter_after_fft,
            )
            # NOTE: we can apply multiplicative factors here that commute with the ramp filter, e.g. a softening function
            out = cufft.irfft(
                self._chunk_mem_horz_filter_after_fft,
                n=self._chunk_mem_horz_filter.shape[1],
                axis=1,
                norm="backward",
                plan=self._cufft_plan_bckward,
            )
            self._e_mem2_free = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(self._e_mem2_free)

            out = out[:, : chunk_shape[1], :]
            out = out[: cparr_view.shape[0], ...]
            return out  # noqa: RET504

    def filter_bulk_axis1_stream(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> typing.Generator[cp.ndarray]:
        """Create a generator of ramp-filtered projection chunks, from an input array."""
        num_slices: int = (
            cparr_view.shape[0] + (self.chunk_shape[0] - 1)
        ) // self.chunk_shape[0]
        for i in range(num_slices):
            sl = slice(
                i * self.chunk_shape[0],
                min((i + 1) * self.chunk_shape[0], cparr_view.shape[0]),
            )
            v = cparr_view[sl]
            out = self._filter_chunk_axis1(v)
            if block:
                self.stream.synchronize()
            yield out

    def filter_bulk_axis1_concat(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> cp.ndarray:
        """Create an output array of ramp-filtered projections, from an input array."""
        out_tup: list[cp.ndarray] = list(
            self.filter_bulk_axis1_stream(cparr_view)
        )
        out = cp.concatenate(out_tup, axis=0)
        if block:
            self.stream.synchronize()
        return out


class FBPFilterer:
    """An object to handle FBP-related filtration operations."""

    rfilt: RampFilterer
    proj_2d_shape: tuple[int, int]  # non-expanded shape
    chunk_shape: tuple[int, int, int]
    L: float
    physpixwid: float
    ss: SamplingStrategy

    _max_abs_rotation_angle: float

    class SamplingStrategy(enum.Enum):
        """Enum for strategies to sample the integral required for the full filtration."""

        SINGLE_ROTATION_RANDOM_UNIFORM = 1

    def __init__(
        self,
        *,
        L: float,  # noqa: N803
        physpixwid: float,
        proj_2d_shape: tuple[int, int],
        projs_per_batch: int = 3,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SINGLE_ROTATION_RANDOM_UNIFORM,
    ) -> None:
        self.L = L
        self.physpixwid = physpixwid
        self.ss = sampling_strategy
        self._max_abs_rotation_angle = self.get_max_abs_rotation_angle(
            proj_2d_shape, physpixwid, L
        )
        self.proj_2d_shape = proj_2d_shape
        self.chunk_shape = (
            projs_per_batch,
            *self.get_expanded_proj_2d_shape(
                proj_2d_shape, self._max_abs_rotation_angle
            ),
        )
        # Construct the ramp-filtering object with memory buffers
        self.rfilt = RampFilterer(
            physpixwid=physpixwid,
            proj_2d_shape=self.chunk_shape[1:],
            projs_per_batch=self.chunk_shape[0],
        )

    @classmethod
    def get_max_abs_rotation_angle(
        cls, proj_2d_shape: tuple[int, int], physpixwid: float, L: float
    ) -> float:
        # Compute the maximum rotation angle.
        # cone_angle_horz: float = 2*np.arctan2(proj_2d_shape[0]*physpixwid/2, L)
        cone_angle_vert: float = 2 * np.arctan2(
            proj_2d_shape[1] * physpixwid / 2,
            np.sqrt(
                np.square(L) + np.square(proj_2d_shape[0] * physpixwid / 2)
            ),
        )
        return cone_angle_vert / 2

    @classmethod
    def get_expanded_proj_2d_shape(
        cls, proj_2d_shape: tuple[int, int], max_abs_rotation_angle: float
    ) -> tuple[int, int]:
        # Compute the expanded projection shape. It needs to expand in the horizontal direction
        # in order to accomodate the rotation.
        sint, cost = (
            np.sin(max_abs_rotation_angle),
            np.cos(max_abs_rotation_angle),
        )
        newwidth = proj_2d_shape[1] * sint + proj_2d_shape[0] * cost
        # Set expanded shape
        expanded_proj_2d_shape = [newwidth, proj_2d_shape[1]]
        # Make them integers
        expanded_proj_2d_shape = [
            int(np.ceil(expanded_proj_2d_shape[0])),
            int(np.ceil(expanded_proj_2d_shape[1])),
        ]
        # Make the difference in size even.
        expanded_proj_2d_shape[0] += (
            expanded_proj_2d_shape[0] - proj_2d_shape[0]
        ) % 2
        expanded_proj_2d_shape[1] += (
            expanded_proj_2d_shape[1] - proj_2d_shape[1]
        ) % 2

        return tuple(expanded_proj_2d_shape)  # ty:ignore[invalid-return-type]

    def _filter_bulk_axis1_stream(
        self,
        cparr_view: cp.ndarray,
        block: bool = True,
        clear_gpu_cache_regularly: bool = True,
    ) -> typing.Generator[cp.ndarray]:
        """Create a generator of filtered projection chunks, from an input array."""
        # Place the rotated data directly into the memory of the rfilt object. This will result in a small optimisation by removing the need for an additional data copy.
        chunkmem = self.rfilt._chunk_mem_horz_filter[
            :, : self.chunk_shape[1], :
        ]
        # num_slices: int = (cparr_view.shape[0] + (self.chunk_shape[0]-1))//self.chunk_shape[0]
        out_buffer = cp.zeros_like(chunkmem)
        event_completed_filter = None
        # Check that input size is correct
        if cparr_view.shape[1:] != self.proj_2d_shape:
            msg = "Wrong input size."
            raise ValueError(msg)
        # Calculate padding amounts to pad the input data before rotating etc
        padx = (self.chunk_shape[1] - self.proj_2d_shape[0]) // 2
        pady = (self.chunk_shape[2] - self.proj_2d_shape[1]) // 2
        for i in itertools.count():
            sl = slice(
                i * self.chunk_shape[0],
                min((i + 1) * self.chunk_shape[0], cparr_view.shape[0]),
            )
            if sl.start >= cparr_view.shape[0]:
                break
            v = cparr_view[sl]
            v = cp.pad(v, ((0, 0), (padx, padx), (pady, pady)))
            # Ensure that the memory is clear before rotating and copying the data in
            if self.rfilt._e_mem1_free is not None:  # noqa: SLF001
                self.rfilt._e_mem1_free.synchronize()  # noqa: SLF001
            # Rotate and copy into the memory buffer
            rotang_rads_list = np.random.uniform(  # noqa: NPY002
                low=-self._max_abs_rotation_angle,
                high=self._max_abs_rotation_angle,
                size=(v.shape[0],),
            )
            events = [None for _ in range(v.shape[0])]
            for proj_index in range(v.shape[0]):
                with cp.cuda.Stream(non_blocking=True) as stream:
                    cupyx.scipy.ndimage.rotate(
                        input=v[proj_index],
                        angle=-180 / np.pi * rotang_rads_list[proj_index],
                        axes=(1, 0),
                        reshape=False,
                        output=chunkmem[proj_index, ...],
                        order=1,
                        mode="grid-constant",
                        cval=0,
                        prefilter=False,  # OK if it blurs a little with prefilter=False for order > 1
                    )
                    events[proj_index] = cp.cuda.Event(
                        block=False, disable_timing=True, interprocess=False
                    )  # Note: block=False ensures that synchronizing the event does not block the CPU thread, only the current GPU stream which waits on the event with e.synchronize()
                    stream.record(events[proj_index])
            # Wait for all rotations to complete
            with self.rfilt.stream as stream:
                for e in events:
                    stream.wait_event(e)
                # Filter the entire chunk at once. The method checks if the chunk is identical to the internal memory buffer and applies a small optimisation if true.
                out = self.rfilt._filter_chunk_axis1(
                    chunkmem, input_in_mem1=True
                )[: v.shape[0], ...]
                event_completed_ramp = cp.cuda.Event(
                    block=False, disable_timing=True, interprocess=False
                )
                stream.record(event_completed_ramp)
            # Before overwriting the output buffer with the next result, yield the previous answer
            if not i == 0:
                if block:
                    # Wait until output array is filled with correct answer before yielding it
                    event_completed_filter.synchronize()  # ty:ignore[unresolved-attribute]
                yield out_buffer
                out_buffer = None
                if clear_gpu_cache_regularly:
                    gpu_tools.clear_cache()
            # Rotate back to original orientation and place into output buffer
            out_buffer = cp.zeros_like(out)
            for proj_index in range(v.shape[0]):
                with cp.cuda.Stream(non_blocking=True) as stream:
                    stream.wait_event(event_completed_ramp)
                    cupyx.scipy.ndimage.rotate(
                        input=out[proj_index],
                        angle=180 / np.pi * rotang_rads_list[proj_index],
                        axes=(1, 0),
                        reshape=False,
                        output=out_buffer[proj_index, ...],
                        order=1,
                        mode="grid-constant",
                        cval=0,
                        prefilter=False,  # OK if it blurs a little with prefilter=False for order > 1
                    )
                    events[proj_index] = cp.cuda.Event(
                        block=False, disable_timing=True, interprocess=False
                    )  # Note: block=False ensures that synchronizing the event does not block the CPU thread, only the current GPU stream which waits on the event with e.synchronize()
                    stream.record(events[proj_index])
            with cp.cuda.Stream(non_blocking=True):
                for e in events:
                    stream.wait_event(e)
                event_completed_filter = cp.cuda.Event(
                    block=True, disable_timing=True, interprocess=False
                )
                stream.record(event_completed_filter)
        if block:
            # Wait until output array is filled with correct answer before yielding it
            event_completed_filter.synchronize()  # ty:ignore[unresolved-attribute]
        yield out_buffer
        out_buffer = None
        if clear_gpu_cache_regularly:
            gpu_tools.clear_cache()

    def _filter_bulk_axis1_concat(
        self, cparr_view: cp.ndarray, clear_gpu_cache_regularly: bool = True
    ) -> cp.ndarray:
        """Create an output array of ramp-filtered projections, from an input array."""
        out_tup: list[cp.ndarray] = list(
            self._filter_bulk_axis1_stream(
                cparr_view,
                block=True,
                clear_gpu_cache_regularly=clear_gpu_cache_regularly,
            )
        )
        out = cp.concatenate(out_tup, axis=0)
        return out  # noqa: RET504


if __name__ == "__main__":
    # Do the test
    ncycles: int = 100
    # filt = RampFilterer(proj_2d_shape=(3001, 4000), projs_per_batch=3, stream=cp.cuda.Stream(non_blocking=True))
    filt = FBPFilterer(
        proj_2d_shape=(4001, 4000), projs_per_batch=1, L=300, physpixwid=0.1
    )
    cparr = cp.zeros((20, 4001, 4000), dtype=np.float32)
    # cparr = cp.random.randn(20, 3001, 4000).astype(cp.float32)
    cparr[:, cparr.shape[1] // 2, :] = 1
    t1 = time.time()
    for _ in range(ncycles):
        x = filt._filter_bulk_axis1_concat(cparr)
        cp.cuda.get_current_stream().synchronize()
    t2 = time.time()
    print(
        f"Milliseconds elapsed per cycle per projection: {1000 * (t2 - t1) / ncycles / cparr.shape[0]:.2f}"
    )
