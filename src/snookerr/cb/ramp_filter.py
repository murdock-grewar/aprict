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
import scipy.fft as fft

from .. import ramp_kernels
from ..lib import gpu_tools

# There is a pesky bug that I can't solve. This flag will apply a hotfix.
_HOTFIX_DISABLE_ASYNCHRONICITY: bool = True


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
    _fft_mults_horz: list[cp.ndarray]
    _chunk_mem_horz_filter: cp.ndarray
    _chunk_mems_horz_filter_after_fft: cp.ndarray

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
        override_kern_funcs: tuple[typing.Callable] | None,
    ) -> None:
        self.chunk_shape = (projs_per_batch, *proj_2d_shape)
        self.physpixwid = physpixwid
        self.stream = (
            cp.cuda.get_current_stream() if stream is None else stream
        )
        self._initialise(
            kern_funcs=override_kern_funcs
            or (ramp_kernels.ramp_kernel_ramlak,)
        )

    @property
    def num_kerns(self) -> int:
        """Return the number of kernels applied by this object."""
        return len(self._fft_mults_horz)

    def _initialise(self, kern_funcs) -> None:  # noqa: ANN001
        with self.stream:
            chunk_shape = self.chunk_shape
            physpixwid = self.physpixwid
            # Find the first fast size for the filtration axis
            h = cufft.next_fast_len(2 * self.chunk_shape[1] - 1)
            self._fft_2d_shape = (h, self.chunk_shape[2])
            # Get the horz kernel as a multiplier in Fourier space
            hk = h + (1 - (h % 2))
            self._fft_mults_horz = []
            for kern_func in kern_funcs:
                kern_horz = kern_func(hk, physpixsize=physpixwid)[:h]
                kern_horz = np.roll(kern_horz, shift=-(hk // 2), axis=0)
                self._fft_mults_horz.append(
                    cp.array(
                        fft.rfft(kern_horz, norm="backward").astype(
                            np.complex128
                        )
                    )
                )
            fft_len = self._fft_mults_horz[0].shape[0]

            # Reserve memory and plans for FFT stuff
            self._chunk_mem_horz_filter = cp.zeros(
                (chunk_shape[0], h, chunk_shape[2]),
                dtype=cp.float32,
            )
            self._cufft_plan_forward = cufft.get_fft_plan(
                self._chunk_mem_horz_filter, axes=(1,), value_type="R2C"
            )

            # Initialise the output arrays
            chunk_mem_horz_filter_after_fft: cp.ndarray = cp.zeros(
                (
                    self._chunk_mem_horz_filter.shape[0],
                    fft_len,
                    self._chunk_mem_horz_filter.shape[2],
                ),
                dtype=cp.complex64,
            )
            self._chunk_mems_horz_filter_after_fft = []
            for _ in kern_funcs:
                self._chunk_mems_horz_filter_after_fft.append(
                    chunk_mem_horz_filter_after_fft.copy()
                )

            self._cufft_plan_bckward = cufft.get_fft_plan(
                chunk_mem_horz_filter_after_fft,
                shape=(self._chunk_mem_horz_filter.shape[1],),
                axes=(1,),
                value_type="C2R",
            )

            self._e_mem1_free = None
            self._e_mem2_free = None

    def _filter_chunk_axis1(
        self,
        cparr_view: cp.ndarray,
        input_in_mem1: bool = False,
        wait_on_events: list[cp.cuda.Event] | None = None,
    ) -> tuple[list[cp.ndarray], cp.cuda.Event]:
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

            if wait_on_events is not None:
                for e in wait_on_events:
                    stream.wait_event(e)
            # Do the horizontal filtration
            # An optimisation: if the input array is the chunk memory, then we don't need to copy it in.
            if input_in_mem1:
                # self._chunk_mem_horz_filter[
                #     : cparr_view.shape[0], chunk_shape[1] :, :
                # ] = 0
                v = self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], chunk_shape[1] :, :
                ]
                cp.multiply(v, 0, out=v)
            else:
                if self._e_mem1_free is not None:
                    self._e_mem1_free.synchronize()
                # self._chunk_mem_horz_filter[
                #     : cparr_view.shape[0], chunk_shape[1] :, :
                # ] = 0
                v = self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], chunk_shape[1] :, :
                ]
                cp.multiply(v, 0, out=v)
                # self._chunk_mem_horz_filter[
                #     : cparr_view.shape[0], : chunk_shape[1], :
                # ] = cparr_view[...]
                v = self._chunk_mem_horz_filter[
                    : cparr_view.shape[0], : chunk_shape[1], :
                ]
                cp.multiply(cparr_view, 1, out=v)
            out_ = cufft.rfft(
                self._chunk_mem_horz_filter,
                axis=1,
                norm="backward",
                overwrite_x=False,  # WARNING: if overwrite_x=True, then out may use the same memory as self._chunk_mem_horz_filter. In that case, the memory is not free until 'out' has been processed, so we can't record the event 'self._e_mem1_free' quite yet. This may have contributed to a difficult-to-diagnose bug in the past. So, best leave this at overwrite_x=False
                plan=self._cufft_plan_forward,
            )
            self._e_mem1_free = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(self._e_mem1_free)

            outs = []
            if self._e_mem2_free is not None:
                self._e_mem2_free.synchronize()
            for index, (fft_mult, chunk_mem_after_fft) in enumerate(
                zip(
                    self._fft_mults_horz,
                    self._chunk_mems_horz_filter_after_fft,
                )
            ):
                cp.multiply(
                    out_,
                    fft_mult[cp.newaxis, :, cp.newaxis],
                    out=chunk_mem_after_fft,
                )
                # NOTE: we can apply multiplicative factors here that commute with the ramp filter, e.g. a softening function
                outs.append(
                    cufft.irfft(
                        chunk_mem_after_fft,
                        n=self._chunk_mem_horz_filter.shape[1],
                        axis=1,
                        norm="backward",
                        plan=self._cufft_plan_bckward,
                    )
                )
            self._e_mem2_free = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(self._e_mem2_free)

            # Create an event to indicate when the filtering operation is complete
            event_filter_complete = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(event_filter_complete)

            for index in range(self.num_kerns):
                outs[index] = outs[index][:, : chunk_shape[1], :]
                outs[index] = outs[index][: cparr_view.shape[0], ...]
            return outs, event_filter_complete

    def filter_bulk_axis1_stream(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> typing.Generator[tuple[list[cp.ndarray], cp.cuda.Event]]:
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
            out, completion_event = self._filter_chunk_axis1(v)
            if block:
                # self.stream.synchronize()
                completion_event.synchronize()
            yield out, completion_event

    def filter_bulk_axis1_concat(
        self, cparr_view: cp.ndarray, block: bool = True
    ) -> tuple[list[cp.ndarray], cp.cuda.Event]:
        """Create an output array of [list of filtered projections, one item for each filter], from an input array."""
        out_gen = self.filter_bulk_axis1_stream(
            cparr_view, block=_HOTFIX_DISABLE_ASYNCHRONICITY
        )
        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            outputs = [
                cp.zeros_like(cparr_view) for _ in range(self.num_kerns)
            ]
            outputs_created_event = stream.record()
        sl_start: int = 0
        events = []
        for index, (out_arrs, e) in enumerate(out_gen):
            sl = slice(sl_start, sl_start + out_arrs[0].shape[0])
            with cp.cuda.Stream(
                non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
            ) as stream:
                stream.wait_event(outputs_created_event)
                stream.wait_event(e)
                for out_index in range(self.num_kerns):
                    cp.multiply(
                        out_arrs[out_index], 1, out=outputs[out_index][sl]
                    )
                ev = cp.cuda.Event(
                    block=_HOTFIX_DISABLE_ASYNCHRONICITY,
                    disable_timing=True,
                    interprocess=False,
                )
                stream.record(ev)
                events.append(ev)
            sl_start += out_arrs[0].shape[0]

        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            for e in events:
                stream.wait_event(e)
            completed_event = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(completed_event)

        if block:
            completed_event.synchronize()

        return outputs, completed_event


class FBPFilterer:
    """An object to handle FBP-related filtration operations."""

    rfilt: RampFilterer
    proj_2d_shape: tuple[int, int]  # non-expanded shape
    chunk_shape: tuple[int, int, int]
    R: float
    L: float
    physpixwid: float
    ss: SamplingStrategy

    _hacky_fan_beam_mode: bool

    event_completed_filter: cp.cuda.Event
    event_completed_ramp: cp.cuda.Event

    _max_abs_rotation_angle: float
    _expanded_2d_shape: tuple[int, int]
    _fft_2d_shape: tuple[int, int]
    _multmask_2d: cp.ndarray

    _include_hilbert_transform: bool

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
        limit_zeta: float
        | None = None,  # limit the maximum rotation angle zeta
        include_hilbert_transform_in_output: bool = False,
        _hacky_fan_beam_mode: bool = False,  # Modifies the behaviour to be compatible with a simple fan-beam
    ) -> None:
        self._include_hilbert_transform = include_hilbert_transform_in_output
        self.hacky_fan_beam_mode = _hacky_fan_beam_mode
        self.L = L
        self.physpixwid = physpixwid
        self.ss = sampling_strategy
        self._max_abs_rotation_angle = self.get_max_abs_rotation_angle(
            proj_2d_shape, physpixwid, L
        )
        if limit_zeta is not None:
            self._max_abs_rotation_angle = min(
                self._max_abs_rotation_angle, limit_zeta
            )
        self.proj_2d_shape = proj_2d_shape
        self._expanded_2d_shape = self.get_expanded_proj_2d_shape(
            proj_2d_shape, self._max_abs_rotation_angle
        )
        # FFT is more efficient with certain sizes.
        self.chunk_shape = (
            projs_per_batch,
            *self._expanded_2d_shape,
        )
        # Construct the ramp-filtering object with memory buffers
        kern_funcs = (ramp_kernels.ramp_kernel_ramlak,)
        if self._include_hilbert_transform:
            kern_funcs += (ramp_kernels.hilbert_transform_kernel,)
        self.rfilt = RampFilterer(
            physpixwid=physpixwid
            / L,  # use the physical pixel width on the plane projected to be 1 unit in front of the source point.
            proj_2d_shape=self.chunk_shape[1:],
            projs_per_batch=self.chunk_shape[0],
            stream=cp.cuda.Stream(
                non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
            ),
            override_kern_funcs=kern_funcs,  # ty:ignore[invalid-argument-type]
        )
        # self._fft_2d_shape = self.rfilt._fft_2d_shape
        self.event_completed_filter = None
        self.event_completed_ramp = None

        # Construct the multiplicative mask
        # NOTE: u is the horizontal coordinate on the detector plane projected to be 1 unit in front of the source point. v is the vertical coordinate.
        uarr = (
            physpixwid
            / L
            * (
                np.arange(self._expanded_2d_shape[0], dtype=np.float32)
                - (self._expanded_2d_shape[0] - 1) / 2
            )
        )
        varr = (
            physpixwid
            / L
            * (
                np.arange(self._expanded_2d_shape[1], dtype=np.float32)
                - (self._expanded_2d_shape[1] - 1) / 2
            )
        )
        self._multmask_2d = cp.array(
            1
            / (
                np.float_power(np.sqrt(1 + np.square(varr[np.newaxis, :])), 3)
                * np.sqrt(
                    1
                    + np.square(varr[np.newaxis, :])
                    + np.square(uarr[:, np.newaxis])
                )
            )
        )

    def get_letterbox_slice(
        self, rotation_angle: float
    ) -> slice[float, float]:
        """Return the slice in fractional indices corresponding to the nonzero parts of the vertical range of the rotated detector."""
        # v*L = physical displacement on detector, along vertical direction.
        # Solve v^2 = sin^2 \theta - cos^2 \theta tan^2 \zeta = sin^2 \theta + cos^2 \theta - cos^2 \theta( 1 + \tan^2 \zeta)
        #       v^2 = 1 - \cos^2 \theta (1 + \tan^2 \zeta)
        #       v^2 = 1 - \cos^2 \theta \sec^2 \zeta
        # Solution: v = +- \sqrt{\cos^2 \theta / \cos^2 \zeta - 1}
        # Then, using v = physpixwid*(index - mid_index)/L, we have
        # index = v*L/physpixwid + mid_index
        # --> index = mid_index +- (L/physpixwid) * sqrt{ ... }
        if abs(rotation_angle) > self._max_abs_rotation_angle:
            msg = "Invalid rotation angle entered."
            raise ValueError(msg)
        mid_index: float = (self._expanded_2d_shape[1] - 1) / 2
        disc: float = (
            self.L
            / self.physpixwid
            * np.sqrt(
                np.maximum(
                    np.square(
                        np.cos(rotation_angle)
                        / np.cos(self._max_abs_rotation_angle),
                    )
                    - 1,
                    0,
                )
            )
        )
        return slice(mid_index - disc, mid_index + disc)

    @property
    def zeta(self) -> float:
        """Get the maximum rotation angle which can be accomodated by this object."""
        return self._max_abs_rotation_angle

    @classmethod
    def get_max_abs_rotation_angle(
        cls,
        proj_2d_shape: tuple[int, int],
        physpixwid: float,
        L: float,  # noqa: N803
    ) -> float:
        """Get the maximum rotation angle which can be accomodated by an instance of this object."""
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
        """Get the 2D shape of the expanded shape of the arrays. They need to be expanded in order to accommodate a rotation."""
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

    def filter_bulk_axis1_stream(  # noqa: C901
        self,
        arr: cp.ndarray | np.ndarray | np.memmap,
        block: bool = True,
        clear_gpu_cache_regularly: bool = False,
        force_rotation_angle: float | None = None,
    ) -> typing.Generator[tuple[list[cp.ndarray], cp.cuda.Event]]:
        """Create a generator of filtered projection chunks, from an input array."""
        # Place the rotated data directly into the memory of the rfilt object. This will result in a small optimisation by removing the need for an additional data copy.
        chunkmem = self.rfilt._chunk_mem_horz_filter[  # noqa: SLF001
            :, : self.chunk_shape[1], :
        ]
        # out_buffer = cp.zeros_like(chunkmem)
        # Check that input size is correct
        if arr.shape[1:] != self.proj_2d_shape:
            msg = "Wrong input size."
            raise ValueError(msg)
        # Calculate padding amounts to pad the input data before rotating etc
        padx = (self._expanded_2d_shape[0] - self.proj_2d_shape[0]) // 2
        pady = (self._expanded_2d_shape[1] - self.proj_2d_shape[1]) // 2
        # padx_fft = (self._fft_2d_shape[0] - self._expanded_2d_shape[0])
        # pady_fft = (self._fft_2d_shape[1] - self._expanded_2d_shape[1])

        # Loop through chunks
        for i in itertools.count():
            sl = slice(
                i * self.chunk_shape[0],
                min((i + 1) * self.chunk_shape[0], arr.shape[0]),
            )
            if sl.start >= arr.shape[0]:
                break
            v = arr[sl]
            if type(arr) is not cp.ndarray:
                v = cp.array(v)  # Convert to a cupy array if necessary
            v = cp.pad(v, ((0, 0), (padx, padx), (pady, pady)))

            # # Ensure that the memory is clear before rotating and copying the data in
            # if self.rfilt._e_mem1_free is not None:
            #     self.rfilt._e_mem1_free.synchronize()

            # Rotate and copy into the memory buffer
            if force_rotation_angle is None:
                rotang_rads_list = np.random.uniform(  # noqa: NPY002
                    low=-self._max_abs_rotation_angle,
                    high=self._max_abs_rotation_angle,
                    size=(v.shape[0],),
                )
            else:
                rotang_rads_list = force_rotation_angle * np.ones(
                    (v.shape[0],), dtype=np.float32
                )
            events = [None for _ in range(v.shape[0])]
            for proj_index in range(v.shape[0]):
                tmp_stream = (
                    cp.cuda.get_current_stream()
                    if _HOTFIX_DISABLE_ASYNCHRONICITY
                    else cp.cuda.Stream(
                        non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
                    )
                )
                with tmp_stream as stream:
                    # The ramp filter uses the same memory we're about to write into.
                    # Wait for it to complete filtering and producing its output before
                    # we overwrite that memory.
                    if self.rfilt._e_mem1_free is not None:  # noqa: SLF001
                        stream.wait_event(self.rfilt._e_mem1_free)  # noqa: SLF001
                    if self.event_completed_ramp is not None:
                        stream.wait_event(self.event_completed_ramp)
                    if i == 0:
                        e = self.event_completed_filter
                        if e is not None:
                            stream.wait_event(e)

                    # HOTFIX: this line should not be necessary! But there are synchronicity issues without it. Bizarre. Unsure if it's a bug with my software or with cupy.
                    # time.sleep(0.05)
                    cp.multiply(
                        chunkmem[proj_index, ...],
                        0,
                        out=chunkmem[proj_index, ...],
                    )
                    # time.sleep(0.05)

                    if rotang_rads_list[proj_index] != 0:
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
                    else:
                        chunkmem[proj_index, ...] = v[proj_index]
                    # Multiply by mask
                    cp.multiply(
                        chunkmem[proj_index, ...],
                        self._multmask_2d,
                        out=chunkmem[proj_index, ...],
                    )
                    # Apply the letterbox
                    # NOTE: THIS IS A PROBBALE SOURCE OF ERROR FOR SMALL ZETA
                    # TO TEST THIS, CONSIDER TESTING THE FAN BEAM RECON BUT ENABLE THE FOLLOWING LOGIC AND SEE WHAT HAPPENS
                    if not self.hacky_fan_beam_mode:
                        sl = self.get_letterbox_slice(
                            rotang_rads_list[proj_index]
                        )
                        # idx_1 = math.floor(sl.start + 1 / 2)
                        # idx_2 = math.ceil(sl.stop - 1 / 2)
                        idx_1 = round(sl.start)
                        idx_2 = round(sl.stop)
                        if idx_1 == idx_2:
                            cp.multiply(
                                chunkmem[proj_index, :, idx_1],
                                sl.stop - sl.start,
                                out=chunkmem[proj_index, :, idx_1],
                            )
                            cp.multiply(
                                chunkmem[proj_index, :, :idx_1],
                                0,
                                out=chunkmem[proj_index, :, :idx_1],
                            )
                            cp.multiply(
                                chunkmem[proj_index, :, idx_1 + 1 :],
                                0,
                                out=chunkmem[proj_index, :, idx_1 + 1 :],
                            )
                        else:
                            # mult_1 = 1 - (sl.start + 1 / 2 - idx_1)
                            mult_1 = (idx_1 + 1 / 2) - sl.start
                            cp.multiply(
                                chunkmem[proj_index, :, idx_1],
                                mult_1,
                                out=chunkmem[proj_index, :, idx_1],
                            )
                            # mult_2 = 1 - (
                            #     idx_2 - (sl.stop - 1 / 2)
                            # )  # TODO: check this  # noqa: FIX002
                            mult_2 = sl.stop - (idx_2 - 1 / 2)
                            cp.multiply(
                                chunkmem[proj_index, :, idx_2],
                                mult_2,
                                out=chunkmem[proj_index, :, idx_2],
                            )
                            cp.multiply(
                                chunkmem[proj_index, :, :idx_1],
                                0,
                                out=chunkmem[proj_index, :, :idx_1],
                            )
                            cp.multiply(
                                chunkmem[proj_index, :, idx_2 + 1 :],
                                0,
                                out=chunkmem[proj_index, :, idx_2 + 1 :],
                            )
                    # Create event to mark completion of the above steps
                    events[proj_index] = cp.cuda.Event(
                        block=_HOTFIX_DISABLE_ASYNCHRONICITY,
                        disable_timing=True,
                        interprocess=True,
                    )  # Note: block=False ensures that synchronizing the event does not block the CPU thread, only the current GPU which waits on the event with e.synchronize()
                    stream.record(events[proj_index])
            # Wait for all rotations to complete
            # with self.rfilt.stream as stream:
            tmp_stream = (
                cp.cuda.get_current_stream()
                if _HOTFIX_DISABLE_ASYNCHRONICITY
                else self.rfilt.stream
            )
            with tmp_stream as stream:
                for e in events:
                    stream.wait_event(e)
                # Filter the entire chunk at once. The method checks if the chunk is identical to the internal memory buffer and applies a small optimisation if true.
                outs, self.event_completed_ramp = (
                    self.rfilt._filter_chunk_axis1(  # noqa: SLF001
                        chunkmem, input_in_mem1=True
                    )
                )
                outs = [arr[: v.shape[0], ...] for arr in outs]
            # Before overwriting the output buffer with the next result, yield the previous answer
            if not i == 0:
                if block:
                    # Wait until output array is filled with correct answer before yielding it
                    self.event_completed_filter.synchronize()
                # yield out_buffer
                yield (
                    [
                        arr[
                            :,
                            padx : ((-padx) or None),
                            pady : ((-pady) or None),
                        ]
                        for arr in outs_buffer  # noqa: F821
                    ],
                    self.event_completed_filter,
                )
                outs_buffer = None
                if clear_gpu_cache_regularly:
                    gpu_tools.clear_cache()
            # Rotate back to original orientation and place into output buffer
            tmp_stream = (
                cp.cuda.get_current_stream()
                if _HOTFIX_DISABLE_ASYNCHRONICITY
                else cp.cuda.Stream(
                    non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
                )
            )
            with tmp_stream as stream:
                outs_buffer = [cp.zeros_like(arr) for arr in outs]
                event_out_buffer_created = stream.record()
            for proj_index in range(v.shape[0]):
                tmp_stream = (
                    cp.cuda.get_current_stream()
                    if _HOTFIX_DISABLE_ASYNCHRONICITY
                    else cp.cuda.Stream(
                        non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
                    )
                )
                with tmp_stream as stream:
                    stream.wait_event(self.event_completed_ramp)
                    stream.wait_event(event_out_buffer_created)
                    # event_completed_ramp.synchronize()
                    for out, out_buffer in zip(outs, outs_buffer):
                        if rotang_rads_list[proj_index] != 0:
                            cupyx.scipy.ndimage.rotate(
                                input=out[proj_index],
                                angle=180
                                / np.pi
                                * rotang_rads_list[proj_index],
                                axes=(1, 0),
                                reshape=False,
                                output=out_buffer[proj_index, ...],
                                order=1,
                                mode="grid-constant",
                                cval=0,
                                prefilter=False,  # OK if it blurs a little with prefilter=False for order > 1
                            )
                        else:
                            out_buffer[proj_index, ...] = out[proj_index]
                    events[proj_index] = cp.cuda.Event(
                        block=_HOTFIX_DISABLE_ASYNCHRONICITY,
                        disable_timing=True,
                        interprocess=False,
                    )  # Note: block=False ensures that synchronizing the event does not block the CPU thread, only the current GPU stream which waits on the event with e.synchronize()
                    stream.record(events[proj_index])
            tmp_stream = (
                cp.cuda.get_current_stream()
                if _HOTFIX_DISABLE_ASYNCHRONICITY
                else cp.cuda.Stream(
                    non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
                )
            )
            with tmp_stream as stream:
                for e in events:
                    stream.wait_event(e)
                self.event_completed_filter = cp.cuda.Event(
                    block=True, disable_timing=True, interprocess=False
                )
                stream.record(self.event_completed_filter)
        if i == 0:
            return
        if block:
            # Wait until output array is filled with correct answer before yielding it
            self.event_completed_filter.synchronize()
        # yield out_buffer
        yield (
            [
                arr[
                    :,
                    padx : ((-padx) or None),
                    pady : ((-pady) or None),
                ]
                for arr in outs_buffer
            ],
            self.event_completed_filter,
        )
        outs_buffer = None
        if clear_gpu_cache_regularly:
            gpu_tools.clear_cache()

    def filter_bulk_axis1_concat(
        self,
        arr: cp.ndarray | np.ndarray | np.memmap,
        clear_gpu_cache_regularly: bool = True,
        force_rotation_angle: float | None = None,
        block: bool = True,
    ) -> tuple[list[cp.ndarray], cp.cuda.Event]:
        """Create an output array of filtered projections, from an input array."""
        out_gen = self.filter_bulk_axis1_stream(
            arr,
            block=_HOTFIX_DISABLE_ASYNCHRONICITY,
            clear_gpu_cache_regularly=clear_gpu_cache_regularly,
            force_rotation_angle=force_rotation_angle,
        )
        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            outputs = [cp.zeros_like(arr) for _ in range(self.rfilt.num_kerns)]
            outputs_created_event = stream.record()
        sl_start: int = 0
        events = []
        for index, (out_arrs, e) in enumerate(out_gen):
            sl = slice(sl_start, sl_start + out_arrs[0].shape[0])
            with cp.cuda.Stream(
                non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
            ) as stream:
                stream.wait_event(e)
                stream.wait_event(outputs_created_event)
                for out_arr, output in zip(out_arrs, outputs):
                    cp.multiply(out_arr, 1, out=output[sl])
                ev = cp.cuda.Event(
                    block=_HOTFIX_DISABLE_ASYNCHRONICITY,
                    disable_timing=True,
                    interprocess=False,
                )
                stream.record(ev)
                events.append(ev)
            sl_start += out_arrs[0].shape[0]

        with cp.cuda.Stream(
            non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY
        ) as stream:
            for e in events:
                stream.wait_event(e)
            completed_event = cp.cuda.Event(
                block=True, disable_timing=True, interprocess=False
            )
            stream.record(completed_event)

        if block:
            completed_event.synchronize()

        return outputs, completed_event


if __name__ == "__main__":
    # Do the test
    ncycles: int = 100
    # filt = RampFilterer(proj_2d_shape=(3001, 4000), projs_per_batch=3, stream=cp.cuda.Stream(non_blocking=not _HOTFIX_DISABLE_ASYNCHRONICITY))
    # shape_2d = (4001, 4000)
    shape_2d = (752, 752)
    filt = FBPFilterer(
        proj_2d_shape=shape_2d, projs_per_batch=3, L=300, physpixwid=0.1
    )
    cparr = cp.zeros((64, *shape_2d), dtype=np.float32)
    cparr = cp.random.randn(*cparr.shape).astype(cp.float32)
    rotangs = np.linspace(-0.1, 0.1, endpoint=True, num=ncycles)
    # cparr[:, cparr.shape[1] // 2, :] = 1
    t1 = time.time()
    completed_events = []
    for _, rotang in zip(range(ncycles), rotangs):
        xs, e = filt.filter_bulk_axis1_concat(
            cparr,
            block=_HOTFIX_DISABLE_ASYNCHRONICITY,
            force_rotation_angle=rotang,
        )
        completed_events.append(e)
    for e in completed_events:
        e.synchronize()
    t2 = time.time()
    print(
        f"Milliseconds elapsed per cycle per projection: {1000 * (t2 - t1) / ncycles / cparr.shape[0]:.2f}"
    )
