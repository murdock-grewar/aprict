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
import math
import multiprocessing
import threading
import time
import typing

import numexpr as ne
import numpy as np
import scipy.ndimage
from scipy import fft

from .. import ramp_kernels
from ..lib import gpu_tools


class FBPFiltererCPU:
    """An object to handle FBP-related filtration operations."""

    proj_2d_shape: tuple[int, int]  # non-expanded shape
    chunk_shape: tuple[int, int, int]
    L: float
    physpixwid: float
    ss: SamplingStrategy

    _max_abs_rotation_angle: float
    _expanded_2d_shape: tuple[int, int]
    _multmask_prerotation_2d_broadcastable: np.ndarray  # 1/sqrt(1 + Y^2/R^2 + X^2/R^2) where Y,X are the horz, vert coords in the detector midplane.
    _multmask_postrotation_vert_1d_broadcastable: np.ndarray

    # Filtering-related allocations
    _fft_mult_horz_broadcastable: np.ndarray
    _chunk_mem_horz_filter: np.ndarray
    _chunk_mem_horz_filter_after_fft: np.ndarray

    _mem_prerotation: np.ndarray
    _mem_postrotation: np.ndarray

    class SamplingStrategy(enum.Enum):
        """Enum for strategies to sample the integral required for the full filtration."""

        SINGLE_ROTATION_RANDOM_UNIFORM = 1

    def __init__(
        self,
        *,
        L: float,  # noqa: N803
        physpixwid: float,
        proj_2d_shape: tuple[int, int],
        projs_per_batch: int | None = None,
        sampling_strategy: SamplingStrategy = SamplingStrategy.SINGLE_ROTATION_RANDOM_UNIFORM,
    ) -> None:
        if projs_per_batch is None:
            projs_per_batch = multiprocessing.cpu_count()
        self.L = L
        self.physpixwid = physpixwid
        self.ss = sampling_strategy
        self._max_abs_rotation_angle = self.get_max_abs_rotation_angle(
            proj_2d_shape, physpixwid, L
        )
        self.proj_2d_shape = proj_2d_shape
        self._expanded_2d_shape = self.get_expanded_proj_2d_shape(
            proj_2d_shape, self._max_abs_rotation_angle
        )
        self.chunk_shape = (
            projs_per_batch,
            *self._expanded_2d_shape,
        )

        # Construct the multiplicative mask
        # NOTE: Y is the horizontal coordinate on the detector. X is the vertical.
        # Yarr = physpixwid*(np.arange(self._expanded_2d_shape[0], dtype=np.float32) - (self._expanded_2d_shape[0]-1)/2)  # noqa: N806
        # Xarr = physpixwid*(np.arange(self._expanded_2d_shape[1], dtype=np.float32) - (self._expanded_2d_shape[1]-1)/2)  # noqa: N806
        Yarr = physpixwid * (
            np.arange(self.proj_2d_shape[0], dtype=np.float32)
            - (self.proj_2d_shape[0] - 1) / 2
        )  # noqa: N806
        Xarr = physpixwid * (
            np.arange(self.proj_2d_shape[1], dtype=np.float32)
            - (self.proj_2d_shape[1] - 1) / 2
        )  # noqa: N806
        self._multmask_prerotation_2d_broadcastable = (
            1
            / (
                np.sqrt(
                    1
                    + np.square(Xarr[np.newaxis, :] / L)
                    + np.square(Yarr[:, np.newaxis] / L)
                )
            )
        )[np.newaxis, :, :].copy()
        self._multmask_postrotation_vert_1d_broadcastable = (
            1 / (np.power(np.sqrt(1 + np.square(Xarr / L)), 3))
        )[np.newaxis, np.newaxis, :].copy()

        # Allocate memory which will be reused.
        self._mem_prerotation = np.zeros(
            (self.chunk_shape[0], *self._expanded_2d_shape), dtype=np.float32
        )
        # The post-rotation memory will be passed to the ramp filter. It is best to use a horizontal length that is efficient for scipy.fft, so we'll expand the horizontal axis appropriately.
        h: int = fft.next_fast_len(self.chunk_shape[1] * 2, real=True)
        self._mem_postrotation = np.zeros(
            (self.chunk_shape[0], h, self.chunk_shape[2]),
            dtype=self._mem_prerotation.dtype,
        )
        self._output_buffer_postrotation = np.zeros_like(self._mem_prerotation)

        # Allocate memory and whatnot for ramp filtering
        hk = (
            h + (1 - h % 2)
        )  # must produce an odd number greater than h to produce the initial kernel
        kern_horz: np.ndarray = (
            ramp_kernels.ramp_kernel_ramlak(
                hk,  # must be odd for this function
                physpixsize=physpixwid,
            )
        )[:h].copy()  # may crop last element
        # Roll the kernel so that the middle coefficient is at the beginning
        kern_horz = np.roll(kern_horz, shift=-(hk // 2), axis=0)
        self._fft_mult_horz_broadcastable = fft.rfft(
            kern_horz, norm="backward"
        )[np.newaxis, :, np.newaxis].copy()

        # Reserve memory and plans for FFT stuff
        # Scipy does not currently implement any method to get FFT plans, even though there is a plan=... argument to fft etc.
        # self._fft_plan_forward = fft.get_fft_plan(
        #     self._chunk_mem_horz_filter, axes=(1,), value_type="R2C"
        # )
        self._chunk_mem_horz_filter_after_fft: np.ndarray = np.zeros(
            (
                self._mem_postrotation.shape[0],
                self._mem_postrotation.shape[1] // 2 + 1,
                self._mem_postrotation.shape[2],
            ),
            dtype=np.complex64,
        )  # Initialise the output array
        # self._fft_plan_bckward = fft.get_fft_plan(
        #     self._chunk_mem_horz_filter_after_fft,
        #     shape=(self._chunk_mem_horz_filter.shape[1],),
        #     axes=(1,),
        #     value_type="C2R",
        # )

    def get_letterbox_slice(
        self, rotation_angle: float
    ) -> slice[float, float]:
        """Return the slice in fractional indices corresponding to the nonzero parts of the vertical range of the rotated detector."""
        # Solve: X^2/L^2 = (cos^2 \theta / cos^2 \zeta - 1)
        # Solution: X/L = +- \sqrt{cos^2 \theta / \cos^2 \zeta - 1}
        # Then, using X = physpixwid*(index-mid_index), we have
        # index = X/physpixwid + mid_index
        # --> index = mid_index +- (L/physpixwid) * sqrt{ ... }
        if abs(rotation_angle) > self._max_abs_rotation_angle:
            msg = "Invalid rotation angle entered."
            raise ValueError(msg)
        mid_index: float = (self._expanded_2d_shape[1] - 1) / 2
        disc: float = (
            self.L
            / self.physpixwid
            * np.sqrt(
                np.square(
                    np.maximum(
                        0,
                        np.cos(rotation_angle)
                        / np.cos(self._max_abs_rotation_angle),
                    )
                )
                - 1
            )
        )
        return slice(mid_index - disc, mid_index + disc)

    def _apply_letterboxing(
        self, rot_ang_rad: float, np_padded_arr_3d: np.ndarray
    ) -> None:
        sl = self.get_letterbox_slice(rot_ang_rad)
        idx_1 = math.floor(sl.start + 1 / 2)
        idx_2 = math.ceil(sl.stop - 1 / 2)
        if idx_1 == idx_2:
            # Multiply the single nonzero row by a number in [0,1] according to how much is included in the letterbox
            vo = np_padded_arr_3d[:, :, idx_1]
            mult = sl.stop - sl.start  # noqa: F841
            ne.evaluate("vo * mult", out=vo)
            # Mask out all other rows
            vo = np_padded_arr_3d[:, :, :idx_1]
            ne.evaluate("0*vo", out=vo)
            vo = np_padded_arr_3d[:, :, idx_1 + 1 :]
            ne.evaluate("0*vo", out=vo)
        else:
            # Multiply the first nonzero row by a number between 0 and 1
            mult_1 = 1 - (sl.start + 1 / 2 - idx_1)  # noqa: F841
            v = np_padded_arr_3d[:, :, idx_1]
            ne.evaluate("v * mult_1", out=v)
            # Multiply the last nonzero row by a number between 0 and 1
            mult_2 = 1 - (
                idx_2 - (sl.stop - 1 / 2)
            )  # TODO: check this  # noqa: F841
            v = np_padded_arr_3d[:, :, idx_2]
            ne.evaluate("v * mult_2", out=v)
            # Mask out the top and bottom
            v = np_padded_arr_3d[:, :, :idx_1]
            ne.evaluate("0*v", out=v)
            v = np_padded_arr_3d[:, :, idx_2 + 1 :]
            ne.evaluate("0*v", out=v)

    @property
    def zeta(self) -> float:
        return self._max_abs_rotation_angle

    @classmethod
    def get_max_abs_rotation_angle(
        cls,
        proj_2d_shape: tuple[int, int],
        physpixwid: float,
        L: float,  # noqa: N803
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
        nparr_view: np.ndarray | np.memmap,
        force_rotation_angle: float | None = None,
    ) -> typing.Generator[np.ndarray]:
        """Create a generator of filtered projection chunks, from an input array."""
        # out_buffer = cp.zeros_like(chunkmem)
        # Check that input size is correct
        if nparr_view.shape[1:] != self.proj_2d_shape:
            msg = "Wrong input size."
            raise ValueError(msg)
        # Calculate padding amounts to pad the input data before rotating etc
        padx = (self.chunk_shape[1] - self.proj_2d_shape[0]) // 2
        pady = (self.chunk_shape[2] - self.proj_2d_shape[1]) // 2

        nparr_view_shape = nparr_view.shape
        tot_num_projs: int = nparr_view_shape[0]

        # Loop through chunks
        for i in itertools.count():
            sl = slice(
                i * self.chunk_shape[0],
                min((i + 1) * self.chunk_shape[0], tot_num_projs),
            )
            if sl.start >= tot_num_projs:
                break

            sl_len = sl.stop - sl.start

            # Instead of directly padding, we will create a new array, and then use numexpr to copy in the data.
            # This is more efficient, since numexpr is multithreaded.
            # v = nparr_view[sl]  # Since this is a contiguous slice, this will return a np.memmap object if nparr_view is a np.memmap
            # v = np.pad(v, ((0, 0), (padx, padx), (pady, pady)))
            # Set the out-of-view parts to 0
            if padx > 0 and pady > 0:
                # Set left to zero
                mem_prerot_outofview = self._mem_prerotation[:sl_len, :padx, :]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
                # Set right to zero
                mem_prerot_outofview = self._mem_prerotation[
                    :sl_len, -padx:, :
                ]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
                # Set top to zero
                mem_prerot_outofview = self._mem_prerotation[
                    :sl_len, padx:-padx, :pady
                ]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
                # Set bottom to zero
                mem_prerot_outofview = self._mem_prerotation[
                    :sl_len, padx:-padx, -pady:
                ]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
            elif padx > 0:
                # Set left to zero
                mem_prerot_outofview = self._mem_prerotation[:sl_len, :padx, :]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
                # Set right to zero
                mem_prerot_outofview = self._mem_prerotation[
                    :sl_len, -padx:, :
                ]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
            elif pady > 0:
                # Set top to zero
                mem_prerot_outofview = self._mem_prerotation[:sl_len, :, :pady]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
                # Set bottom to zero
                mem_prerot_outofview = self._mem_prerotation[
                    :sl_len, :, -pady:
                ]
                ne.evaluate("0*mem_prerot_outofview", out=mem_prerot_outofview)
            nparr_sliced_view = nparr_view[sl]  # noqa: F841
            # print("padx: ", padx)
            mem_prerot_view = self._mem_prerotation[
                :sl_len, padx : ((-padx) or None), pady : ((-pady) or None)
            ]
            # print("mem_prerot_view shape: ", mem_prerot_view.shape)
            prerot_factor = self._multmask_prerotation_2d_broadcastable  # noqa: F841
            ne.evaluate(
                "nparr_sliced_view * prerot_factor", out=mem_prerot_view
            )  # multi-threaded copy of chunk into 'v'. At the same time, apply the pre-filter multiplication factor, which commutes with the rotation.

            # Rotations are not multithreaded, so we'll launch a separate thread for each layer we want to rotate.
            def do_inverse_rotation(layer_index: int, rot_ang: float) -> None:
                scipy.ndimage.rotate(
                    input=ramp_filtered_data[
                        layer_index, : self._mem_prerotation.shape[1], :
                    ],
                    angle=180 / np.pi * rot_ang,
                    axes=(1, 0),
                    reshape=False,
                    output=self._output_buffer_postrotation[layer_index, ...],
                    order=1,
                    mode="grid-constant",
                    cval=0,
                    prefilter=False,  # Has no effect when order = 1. OK if it blurs a little with prefilter=False for order > 1
                )

            def do_rotation_then_letterbox(
                layer_index: int, rot_ang: float
            ) -> None:
                """Do a rotation on the given layer index, at a given rotation angle. Outputs a 2D array."""
                if True:
                    return
                scipy.ndimage.rotate(
                    input=self._mem_prerotation[layer_index],
                    angle=-180 / np.pi * rot_ang,
                    axes=(1, 0),
                    reshape=False,
                    output=self._mem_postrotation[
                        layer_index, : self._mem_prerotation.shape[1], :
                    ],
                    order=1,
                    mode="grid-constant",
                    cval=0,
                    prefilter=False,  # Has no effect when order = 1. OK if it blurs a little with prefilter=False for order > 1
                )
                self._apply_letterboxing(
                    rot_ang, self._mem_postrotation[[layer_index]]
                )

            # Determine the rotatian angles. It could be different angles for each layer, or a uniform rotation.
            # Rotate and copy into the memory buffer
            if force_rotation_angle is None:
                rotang_rads_list = np.random.uniform(  # noqa: NPY002
                    low=-self._max_abs_rotation_angle,
                    high=self._max_abs_rotation_angle,
                    size=(sl_len,),
                )
            else:
                rotang_rads_list = force_rotation_angle * np.ones(
                    (sl_len,), dtype=np.float32
                )
            # Clear the post-rotation memory
            v = self._mem_postrotation
            ne.evaluate(
                "0*v", out=self._mem_postrotation
            )  # clear the post-rotation memory
            # Launch threads to rotate each layer and apply the letterboxing to each layer
            threads = []
            for proj_index, rot_ang in enumerate(rotang_rads_list):
                t = threading.Thread(
                    target=do_rotation_then_letterbox,
                    args=(proj_index, rot_ang),
                )
                threads.append(t)
                t.start()
            [t.join() for t in threads]

            # Apply the post-rotation multiplier. This commutes withe the ramp filter.
            m = self._multmask_postrotation_vert_1d_broadcastable  # noqa: F841
            v = self._mem_postrotation[:sl_len]
            ne.evaluate("v * m", out=v)

            ## Apply the ramp filter
            fft_out = fft.rfft(
                self._mem_postrotation[:sl_len, ...],
                axis=1,
                norm="backward",
                overwrite_x=True,  # allow destruction of data inside the input array, if that's more efficient
                workers=sl_len,  # match number of workers to CPU count
                # plan=self._fft_plan_forward,
            )
            fft_mult = self._fft_mult_horz_broadcastable  # noqa: F841
            ne.evaluate("fft_out * fft_mult", out=fft_out)
            ramp_filtered_data = fft.irfft(
                fft_out,
                n=self._mem_postrotation.shape[1],
                axis=1,
                norm="backward",
                overwrite_x=True,  # allow destruction of data inside the input array, if that's more efficient
                workers=sl_len,
                # plan=self._fft_plan_bckward,
            )

            # Finally, rotate the projections back to their original orientations
            # Launch threads to rotate each layer and apply the letterboxing to each layer
            threads = []
            for proj_index, rot_ang in enumerate(rotang_rads_list):
                t = threading.Thread(
                    target=do_inverse_rotation, args=(proj_index, rot_ang)
                )
                threads.append(t)
                t.start()
            [t.join() for t in threads]

            # Copy to an appropriately-sized array with numexpr
            # This makes concatenating faster.
            output = np.zeros(
                (
                    sl_len,
                    self._output_buffer_postrotation.shape[1] - 2 * padx,
                    self._output_buffer_postrotation.shape[2] - 2 * pady,
                ),
                dtype=self._output_buffer_postrotation.dtype,
            )
            out_view = self._output_buffer_postrotation[  # noqa: F841
                :sl_len,
                padx : ((-padx) or None),
                pady : ((-pady) or None),
            ]
            ne.evaluate("out_view", out=output)
            yield output

    def _filter_bulk_axis1_concat(
        self,
        nparr_view: np.ndarray | np.memmap,
        force_rotation_angle: float | None = None,
    ) -> np.ndarray:
        """Create an output array of ramp-filtered projections, from an input array."""
        out_arrs: list[np.ndarray] = list(
            self._filter_bulk_axis1_stream(
                nparr_view,
                force_rotation_angle=force_rotation_angle,
            )
        )
        return np.concatenate(out_arrs, axis=0)


if __name__ == "__main__":
    # Do the test
    ncycles: int = 100
    # filt = RampFilterer(proj_2d_shape=(3001, 4000), projs_per_batch=3, stream=cp.cuda.Stream(non_blocking=True))
    # shape_2d = (4001, 4000)
    shape_2d = (752, 752)
    filt = FBPFiltererCPU(
        proj_2d_shape=shape_2d, projs_per_batch=None, L=300, physpixwid=0.1
    )
    nparr = np.zeros((64, *shape_2d), dtype=np.float32)
    nparr = np.random.randn(*nparr.shape).astype(np.float32)
    rotangs = np.linspace(-0.1, 0.1, endpoint=True, num=ncycles)
    # nparr[:, nparr.shape[1] // 2, :] = 1
    t1 = time.time()
    for _, rotang in zip(range(ncycles), rotangs):
        x = filt._filter_bulk_axis1_concat(nparr, force_rotation_angle=rotang)
    t2 = time.time()
    print(
        f"Milliseconds elapsed per cycle per projection: {1000 * (t2 - t1) / ncycles / nparr.shape[0]:.2f}"
    )
