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

"""A file to implement the volume filtration corresponding to the softened Colsher window. This is copied from the legacy `apricot` module."""

import logging
import typing

import cupy as cp
import cupyx.scipy.fft as cufft
import cupyx.scipy.ndimage
import numexpr as ne
import numpy as np
import scipy.fft
import scipy.special

from .. import lib

lib.gpu_tools.cupy_set_fft_plan_cache_size(0)  # cupy is extremely wasteful of VRAM and does not pay attention to memory limits. This setting will limit the VRAM runaway from computing ffts.

# Let 'theta' be the polar angle of the frequency component (kx,ky,kz), measured from the positive kz axis.
# When theta = 0, the orthogonal ring (of unit radius) is fully contained in the region described by (kz^2/(kx^2 + ky^2) < tan(cone_angle/2)^2), and the usual ramp filter applies.
# As Theta moves off-axis, and becomes greater than cone_angle/2, then part of the ring no longer interesects that region.
# The fourier multiplier, which is |k| for the ramp filter, must be divided by the portion of the ring (out of 1) which is contained in the region.
# Here, we calculate that portion as a function of Theta and the cone_angle.
# That portion is equal to (1 - [the arc length between the two intersection points of the ring with the top of the region]/pi).
# Therefore, we wish to calculate the angles of intersection between the ring of unit radius, and the region.
# (kz^2/(kx^2 + ky^2)) = tan(cone_angle/2)^2)
# kz = sin(theta)cos(phi), and kx^2 + ky^2 = sin(phi)^2 + cos(phi)^2 cos(theta)^2
# --> sin(theta)^2 cos(phi)^2 / (sin(phi)^2 + cos(phi)^2 cos(theta)^2) = tan(cone_angle/2)^2
# --> sin(theta)^2 cos(phi)^2 / (1 + (cos(theta)^2 - 1)*cos(phi)^2) = tan(cone_angle/2)^2
# --> sin(theta)^2 cos(phi)^2 / (1 - sin(theta)^2 cos(phi)^2) = tan(cone_angle/2)^2
# --> -1 + 1 / (1 - sin(theta)^2 cos(phi)^2) = tan(cone_angle/2)^2
# --> 1 / (1 - sin(theta)^2 cos(phi)^2) = 1 + tan(cone_angle/2)^2
# --> 1 - sin(theta)^2 cos(phi)^2 = 1/(1 + tan(cone_angle/2)^2)
# --> sin(theta)^2 cos(phi)^2 = 1 - 1/(1 + tan(cone_angle/2)^2)
# --> cos(phi)^2 = (1 - 1/(1 + tan(cone_angle/2)^2))/sin(theta)^2
# --> cos(phi) = sqrt(1 - 1/(1 + tan(cone_angle/2)^2)) / sin(theta)
# --> phi_1 =   arccos(sqrt(1 - 1/(1 + tan(cone_angle/2)^2)) / sin(theta))
# --> phi_2 = - arccos(sqrt(1 - 1/(1 + tan(cone_angle/2)^2)) / sin(theta))
# --> |phi_1 - phi_2| = 2 arccos(
#   sqrt(
#       1 - 1/(1 + tan(cone_angle/2)^2)
#   ) / abs(sin(theta))
# )
# Needed to put abs(...) on sin(theta).
def _conebeam_kfactor_unchunked(theta_k_el, cone_angle, theta_soft):

    tp = cp if type(theta_k_el) is cp.ndarray else np

    if cone_angle == np.pi:
        if theta_soft != 0:
            raise NotImplementedError()
        return 2 * np.pi

    if tp is cp:
        cp.abs(theta_k_el, out=theta_k_el)
        mask = tp.logical_and(
            (cone_angle/2 - theta_soft) < theta_k_el,
            theta_k_el < tp.pi - (cone_angle/2 - theta_soft),
        )
        out = tp.empty_like(theta_k_el)
        out[~mask] = 2*tp.pi
        tkm = theta_k_el[mask]
        # Scalars
        gamma = cp.divide(1, cp.sin(cone_angle/2-theta_soft) - cp.sin(cone_angle/2))
        chi   = cp.multiply(gamma, cp.sin(cone_angle/2))
        # theta_k_el dependent
        sintkm = cp.sin(tkm)
        if cone_angle/2 - theta_soft <= 0:
            alpha = 0
        else:
            alpha = cp.maximum(0, cp.minimum(1, cp.divide(cp.sin(cone_angle/2 - theta_soft), sintkm)))
        beta  = cp.maximum(0, cp.minimum(1, cp.where(cp.sin(cone_angle/2) >= sintkm, 1, cp.divide(cp.sin(cone_angle/2), sintkm))))
        term1 = cp.arcsin(beta) - cp.arcsin(alpha)
        term2 = cp.sqrt(1 - cp.square(beta))
        term3 = cp.sqrt(1 - cp.square(alpha))
        # Cubic equation in sin(theta_k_el). Add terms one-by-one.
        # out_masked = a + bx + cx^2 + dx^3
        # Add term 'a'
        out[mask] = (
            term1 * 4 * cp.square(chi) * (3 + 2*chi) + 4 * cp.arcsin(alpha))
        # Add term 'bx'
        out[mask] += (
            sintkm            * (24 * gamma * chi * (1 + chi) * (term2 - term3)))
        out[mask] += (
            cp.square(sintkm) * (6 * cp.square(gamma) * (1 + 2 * chi) * (term1 - beta*term2 + alpha*term3)))
        out[mask] += (
            cp.float_power(sintkm, 3) * (
                -2/3 * cp.float_power(gamma, 3) * (
                    -9 * (term2 - term3) +
                    (cp.cos(3*cp.arcsin(beta)) - cp.cos(3*cp.arcsin(alpha)))
                )
            )
        )
        return out
    else:
        theta_k_el = ne.evaluate('abs(theta_k_el)', out=theta_k_el)
        mask = tp.logical_and(
            (cone_angle/2 - theta_soft) < theta_k_el,
            theta_k_el < tp.pi - (cone_angle/2 - theta_soft),
        )
        out = tp.empty_like(theta_k_el)
        out[~mask] = 2*tp.pi
        tkm = theta_k_el[mask]
        out_masked = out[mask]
        # Scalars
        gamma = ne.evaluate('1/(sin(cone_angle/2-theta_soft) - sin(cone_angle/2))')
        chi   = ne.evaluate('sin(cone_angle/2) * gamma')
        # theta_k_el dependent
        sintkm = ne.evaluate('sin(tkm)')
        if cone_angle/2 - theta_soft <= 0:
            alpha = 0
        else:
            alpha = ne.evaluate('maximum(0, minimum(1, sin(cone_angle/2 - theta_soft) / sintkm))')
        beta  = ne.evaluate('maximum(0, minimum(1, where(sin(cone_angle/2) >= sintkm, 1, sin(cone_angle/2) / sintkm)))')
        term1 = ne.evaluate('arcsin(beta) - arcsin(alpha)')
        term2 = ne.evaluate('sqrt(1 - beta**2)')
        term3 = ne.evaluate('sqrt(1 - alpha**2)')
        # Cubic equation in sin(theta_k_el). Add terms one-by-one.
        # out_masked = a + bx + cx^2 + dx^3
        # Add term 'a'
        ne.evaluate('term1 * 4 * (chi*chi) * (3 + 2*chi) + 4 * arcsin(alpha)', out=out_masked)
        # Add term 'bx'
        ne.evaluate('out_masked + sintkm    * (24 * gamma * chi * (1 + chi) * (term2 - term3))', out=out_masked)
        ne.evaluate('out_masked + sintkm**2 * (6 * gamma**2 * (1 + 2 * chi) * (term1 - beta*term2 + alpha*term3))', out=out_masked)
        ne.evaluate('''out_masked + sintkm**3 * (
                        -2/3 * gamma**3 * (
                            -9 * (term2 - term3) +
                            (cos(3*arcsin(beta)) - cos(3*arcsin(alpha)))
                        )
                    )''', out=out_masked)
        out[mask] = out_masked # Don't know why this is necessary
        ne.evaluate('out', out=out)
        return out

def _conebeam_kfactor(theta_k_el, cone_angle, theta_soft, mb_per_chunk: typing.Optional[int] = 128, allow_overwrite_theta_k_el: bool = False):

    # NOTE: the mb_per_chunk option is only applied if the array is 3-dimensional.
    # This method is only designed for input arrays `theta_k_el` of dimension 1 or 2.

    tp = cp if type(theta_k_el) is cp.ndarray else np
    if allow_overwrite_theta_k_el:
        out = theta_k_el
    else:
        out = tp.empty_like(theta_k_el)

    if mb_per_chunk is None or theta_k_el.ndim == 2:
        out[...] = _conebeam_kfactor_unchunked(theta_k_el, cone_angle, theta_soft)
        return out

    def slice_gen(slicelen: int):
        step:   int = slicelen
        zstart: int = 0
        zstop:  int = step
        while zstart < out.shape[2]:
            yield slice(zstart, min(zstop, out.shape[2]))
            zstart += step
            zstop  += step
    numzperpart = max(
        1,
        int(np.floor(
        mb_per_chunk * 1024**2 / (out.shape[0]*out.shape[1]*4) # 4 because there are four bytes in a 32-bit float, and I'm assuming that the datatype is cp.float32
        ))
    )
    for sl in slice_gen(numzperpart):
        Z_sub_arr = out[... , sl]
        Z_sub_arr[...] = _conebeam_kfactor_unchunked(theta_k_el[... , sl], cone_angle, theta_soft)
    return out


def _conebeam_kfactor_unchunked_unweighted_approx(theta_k_el, cone_angle, theta_soft):
    """theta_soft currently assumed to be 0."""
    tp = cp if type(theta_k_el) is cp.ndarray else np
    if tp is cp:
        raise NotImplementedError("This method has not been implemented for cupy arrays.")
    else:
        pi = np.pi
        theta_k = ne.evaluate('abs(pi/2 - abs(theta_k_el))', out=theta_k_el, casting='same_kind')
        del theta_k_el
        out = tp.empty_like(theta_k)
        # Scalars
        sintkm = ne.evaluate('sin(theta_k)')
        U = ne.evaluate('where(sin(cone_angle/2) >= sintkm, 1, sin(cone_angle/2) / sintkm)')
        # term = tp.empty_like(U)
        # Contribution from second term
        ne.evaluate('tan(theta_k)**2 * U * sqrt(2*(1-U**2))', out=out, casting='same_kind')
        ne.evaluate('- out / sqrt(2+(U**2)*(-1+cos(2*theta_k)))', out=out)
        # Contribution from first term
        term = ne.evaluate('1/(cos(theta_k)**2)', out=theta_k, casting='same_kind') # Overwrite theta_k
        sintkmsq = ne.evaluate('sintkm**2', out=sintkm) # Overwrite sintkm
        del sintkm
        mask = U == 1
        term[ mask] *= scipy.special.ellipe(sintkmsq[ mask])
        term[~mask] *= scipy.special.ellipeinc(np.arcsin(U[~mask]), sintkmsq[~mask])
        out += term
        ne.evaluate('out * 4', out=out)
        # Handle the exceptional case where theta_k = pi/2.
        mask = sintkmsq >= 1 # Where theta_k = pi/2
        Umask = U[mask] # NOTE: numexpr doesn't work with boolean masks! It fails silently!
        # ne.evaluate('log((1+Umask)/(1-Umask)) - 2*Umask/(Umask**2 - 1)', out=out[mask]) # There is a factor of 1/4 which is cancelled by multiplying with 4.
        out[mask] = np.log((1+Umask)/(1-Umask)) - 2*Umask/(np.square(Umask) - 1)
        del pi
        return out


def _conebeam_kfactor_unweighted_approx(theta_k_el, cone_angle, theta_soft, mb_per_chunk: typing.Optional[int] = 128,
                      allow_overwrite_theta_k_el: bool = False):

    tp = cp if type(theta_k_el) is cp.ndarray else np

    if allow_overwrite_theta_k_el:
        out = theta_k_el
    else:
        out = tp.empty_like(theta_k_el)


    if mb_per_chunk is None:
        out[...] = _conebeam_kfactor_unchunked_unweighted_approx(theta_k_el, cone_angle, theta_soft)
        return out

    def slice_gen(slicelen: int):
        step: int = slicelen
        zstart: int = 0
        zstop: int = step
        while zstart < out.shape[2]:
            yield slice(zstart, min(zstop, out.shape[2]))
            zstart += step
            zstop += step

    numzperpart = max(
        1,
        int(np.floor(
            mb_per_chunk * 1024 ** 2 / (out.shape[0] * out.shape[1] * 4)
            # 4 because there are four bytes in a 32-bit float, and I'm assuming that the datatype is cp.float32
        ))
    )
    for sl in slice_gen(numzperpart):
        Z_sub_arr = out[... , sl]
        Z_sub_arr[...] = _conebeam_kfactor_unchunked_unweighted_approx(theta_k_el[... , sl], cone_angle, theta_soft)
    return out


def ramp_filter_conebeam(
        # arr: np.ndarray | cp.ndarray, # (lenZ, lenX, lenY)
        arr_container: lib.np_tools.NDArrayContainer,
        voxel_phys_len: float, # It needs to know the length of a voxel because the output of the ramp filter is (inversely?) proportional to it.
        cone_angle: float,
        theta_soft: float,
        auto_dc_adjust: bool,
        logger: logging.Logger | None = None,
        log_prefix: str | None = None,
        allow_overwrite_arr: bool = False,
        lowres_correction_array: np.ndarray | None = None, # This is added to the result, BEFORE adjusting the DC offset.
        softener: np.ndarray | cp.ndarray | None = None,
        copy_to_array: np.ndarray | cp.ndarray | None = None,
        extra_multiplier: float = 1,
        ) -> np.ndarray | cp.ndarray:

    if logger is None:
        logger = logging.Logger("")

    if cone_angle == 0:
        logger.warning("" if log_prefix is None else log_prefix + (
        "ramp_filter_conebeam has been supplied with cone_angle = 0. This is not intended use. This may work, but it will not work well because the padding required is very large in the x and y directions."))

    # Set up the overall multiplier -- ensures that the output is the same regardless of the resolution. Use the 'extra_multiplier' to multiply the result by a different amount, e.g. to add back in the factor of voxel-length.
    multiplier = extra_multiplier / voxel_phys_len

    if lowres_correction_array is not None:
        shap  = arr_container.array.shape
        shapl = lowres_correction_array.shape
        if (
            (shap[0]%shapl[0] != 0) or
            (shap[1]%shapl[1] != 0) or
            (shap[2]%shapl[2] != 0)
        ):
            raise ValueError("The provided lowres_correction_array shape does not divide into the full array shape.")
        df1 = shap[0]//shapl[0]
        df2 = shap[1]//shapl[1]
        df3 = shap[2]//shapl[2]
        if (df1 != df2) or (df2 != df3) or (df1 != df3):
            raise ValueError("The provided lowres_correction_array shape does not divide evenly into the full array shape.")
        downfactor: int = df1

    if cone_angle < 0:
        raise ValueError(f"Cone angle is negative: {cone_angle}")
    if cone_angle > np.pi:
        raise ValueError(f"Cone angle greater than pi: {cone_angle}")
    if softener is not None and type(arr_container.array) is not type(softener):
        raise ValueError("The type of 'arr' and of 'softener' do not match.")
    if copy_to_array is not None:
        if copy_to_array.shape != arr_container.array.shape:
            raise ValueError(f"Output array does not match the shape of the input array.")

    input_array_is_cupy = type(arr_container.array) is cp.ndarray
    tp  = cp    if input_array_is_cupy else np
    fft = cufft if input_array_is_cupy else scipy.fft
    # signal = cupyx.scipy.signal if input_array_is_cupy else scipy.signal

    arr_shape = arr_container.array.shape # Will copy-by-value, since arr.shape is an immutable type (Tuple).

    if auto_dc_adjust:
        # Create index masks that indicate where the backprojection is zero. Only sample the top and bottom z layers.
        topzmask = tp.nonzero(arr_container.array[..., 0] == 0)
        botzmask = tp.nonzero(arr_container.array[...,-1] == 0)
        numels = len(topzmask[0]) + len(botzmask[0])
        # approx_num_layers_zeros = numels / arr_shape[0] / arr_shape[1]
        if numels == 0:
            logger.warning(
                ("" if log_prefix is None else log_prefix) +
                f"There were no zero-valued backprojections in either the top or bottom Z slice. The auto-DC adjustment will not be able to complete."
            )
        del numels

    if log_prefix is not None:
        logger.debug(log_prefix + f"Computing forward DFT...")
    if tp is cp:
        f_tparr = fft.rfftn(arr_container.array, norm='ortho', overwrite_x=allow_overwrite_arr)
    else:
        f_tparr = fft.rfftn(arr_container.array, norm='ortho', overwrite_x=allow_overwrite_arr, workers=-1)
    if allow_overwrite_arr:
        if log_prefix is not None:
            logger.debug(log_prefix + f"Freeing memory of the input array...")
        arr_container.array = None # If this was the last reference to this array, then the array should now be garbage-collected.
        # If it was a CuPy array, then clear the cache now so that memory is freed.
        if input_array_is_cupy:
            lib.gpu_tools.clear_cache()
    if log_prefix is not None:
        logger.debug(log_prefix + f"Applying ramp filter & extra factor for cylindrical geometry...")
    freqs_0 = fft.fftfreq(n=arr_shape[0])[:, tp.newaxis, tp.newaxis]
    freqs_1 = fft.fftfreq(n=arr_shape[1])[tp.newaxis, :, tp.newaxis]
    freqs_2 = fft.rfftfreq(n=arr_shape[2])[tp.newaxis, tp.newaxis, :]
    def slice_gen(slicelen: int):
        nonlocal f_tparr
        step:   int = slicelen
        zstart: int = 0
        zstop:  int = step
        while zstart < f_tparr.shape[2]:
            yield slice(zstart, min(zstop, f_tparr.shape[2]))
            zstart += step
            zstop  += step
    numzperpart = max(
        1,
        int(np.floor(
        256 # Hardcoded at 256 MB.
        * 1024**2 / (f_tparr.shape[0]*f_tparr.shape[1]*4) # 4 because there are four bytes in a 32-bit float, and I'm assuming that the datatype is cp.float32
        ))
    )
    pi = np.pi
    if input_array_is_cupy:
        lateral_k_mag_sq = tp.square(freqs_0) + tp.square(freqs_1)
    else:
        lateral_k_mag_sq = ne.evaluate('freqs_0**2 + freqs_1**2')
    for sl in slice_gen(numzperpart):
        freqs_2_slice = freqs_2[..., sl]
        f_tparr_slice = f_tparr[..., sl]
        if cone_angle == 0:
            lateral_k_mag = tp.sqrt(lateral_k_mag_sq)
            if input_array_is_cupy:
                f_tparr[..., sl] *= lateral_k_mag * multiplier
            else:
                ne.evaluate('f_tparr_slice * lateral_k_mag * multiplier', out=f_tparr_slice, casting='same_kind')
            extra_factor = None # For the `del extra_factor` statement later
        else:
            if input_array_is_cupy:
                f_tparr[..., sl] *= tp.sqrt(lateral_k_mag_sq + tp.square(freqs_2_slice)) * multiplier # Multiply by ramp filter
                theta_el = tp.abs(tp.pi/2 - tp.arctan2(freqs_2_slice, tp.sqrt(lateral_k_mag_sq)))
            else:
                ne.evaluate('f_tparr_slice * sqrt(lateral_k_mag_sq + freqs_2_slice**2) * multiplier', out=f_tparr_slice, casting='same_kind')
                theta_el = ne.evaluate('abs(pi/2 - arctan2(freqs_2_slice, sqrt(lateral_k_mag_sq)))')
            # extra_factor = _conebeam_kfactor(theta_el, cone_angle, theta_soft, mb_per_chunk=1024, allow_overwrite_theta_k_el=True)
            if cone_angle == np.pi:
                extra_factor = 2 * np.pi
            else:
                extra_factor = _conebeam_kfactor(theta_el, cone_angle, theta_soft, mb_per_chunk=None, allow_overwrite_theta_k_el=True)
            if input_array_is_cupy:
                f_tparr[..., sl] /= extra_factor
            else:
                ne.evaluate('f_tparr_slice / extra_factor', out=f_tparr_slice, casting='same_kind')
    del pi
    del lateral_k_mag_sq
    del extra_factor
    if log_prefix is not None:
        logger.debug(log_prefix + f"Adding the discretisation-correction...")
    f_tparr *= tp.sinc(freqs_0)
    f_tparr *= tp.sinc(freqs_1)
    f_tparr *= tp.sinc(freqs_2)

    f_tparr[0, 0, 0] = 0

    if log_prefix is not None:
        logger.debug(log_prefix + f"Computing inverse DFT...")
    if input_array_is_cupy:
        # CuPy
        out = fft.irfftn(f_tparr, s=arr_shape, norm='ortho', axes=tuple(range(f_tparr.ndim)), overwrite_x=allow_overwrite_arr)
    else:
        # NumPy / SciPy
        out = fft.irfftn(f_tparr, s=arr_shape, norm='ortho', axes=tuple(range(f_tparr.ndim)), overwrite_x=allow_overwrite_arr, workers=-1)
    del f_tparr # Remove reference to the Fourier transform now. It should be garbage-collected.

    if lowres_correction_array is not None:
        cp.cuda.get_current_stream().synchronize() # Prevent log messages from being printed prematurely
        if log_prefix is not None:
            logger.debug(log_prefix + f"Applying low-pad correction...")
        def slice_gen(slicelen: int):
            step:   int = slicelen
            zstart: int = 0
            zstop:  int = step
            while zstart < out.shape[2]:
                yield slice(zstart, min(zstop, out.shape[2]))
                zstart += step
                zstop  += step

        # I don't know why it's so VRAM-hungry.
        MB_VRAM_avail_tot = lib.gpu_tools.get_gpu_mb_used_and_total()[1]
        MB_per_chunk: float = 512 / 24000 * MB_VRAM_avail_tot # Unfortunately, the use of meshgrid mean this memory choice may be quadrupled or even more
        slicelen = max(1, int(np.floor(
            MB_per_chunk / out.shape[0] / out.shape[1] / np.dtype(np.float32).itemsize * 1024**2
        )))
        slicelen = max(1, int(np.floor(slicelen/downfactor))) * downfactor # Make slicelen a multiple of downfactor

        # Do it with CuPy, regardless of input type.
        xc = cp.arange(out.shape[0])/downfactor + 0//downfactor - 1/2 + 1/downfactor/2
        yc = cp.arange(out.shape[1])/downfactor + 0//downfactor - 1/2 + 1/downfactor/2

        if type(lowres_correction_array) is np.ndarray:
            lowres_correction_array_sliced_np = cupyx.empty_pinned(
                (lowres_correction_array.shape[0], lowres_correction_array.shape[1], slicelen+2), dtype=np.float32)
        lowres_correction_array_sliced_cp = cp.empty(
            (lowres_correction_array.shape[0], lowres_correction_array.shape[1], slicelen+2), dtype=np.float32)

        add_np = None
        add_cp = None
        for sl in slice_gen(slicelen):
            zc = cp.arange((sl.stop - sl.start))/downfactor + sl.start//downfactor - 1/2 + 1/downfactor/2
            # Note on CuPy:
            # CuPy (and I think scipy too) has a bug: the 'mode = whatever' is supposed to *extend the input array*, but it doesn't have this effect. If you try to fetch an interpolated value out of bounds with 'mode = constant', then it will give zero, and NOT the interpolation between 0 and the edge voxel as it should. For this reason, we need to take padded slices of the array.
            # Note: I have since learned that 'mode='grid-constant'' does what I want. Oh well, not changing the code now.
            downslice = slice(sl.start//downfactor, sl.stop//downfactor) # sl is a multiple of downfactor
            downslice_padded = slice(max(0, downslice.start-1), min(lowres_correction_array.shape[2], downslice.stop+1)) # Pad by 1 on either side, so that the interpolation of chunks doesn't fall off the edge
            if type(lowres_correction_array) is cp.ndarray:
                lowres_correction_array_sliced_cp[... , :downslice_padded.stop-downslice_padded.start] = lowres_correction_array[... , downslice_padded]
            else:
                lib.ne_tools.fastcopy(
                    dst=lowres_correction_array_sliced_np[... , :downslice_padded.stop-downslice_padded.start],
                    src=lowres_correction_array[ ... , downslice_padded ])
                lowres_correction_array_sliced_cp.set(lowres_correction_array_sliced_np)
            zc -= downslice_padded.start # z indices should start at zero where the sliced array begins.
            mg = cp.array(cp.meshgrid(xc, yc, zc, indexing='ij'))
            # mg = cp.meshgrid(xc, yc, zc, indexing='ij', copy=False)
            add_cp = cupyx.scipy.ndimage.map_coordinates(
                lowres_correction_array_sliced_cp[:, :, :(downslice_padded.stop - downslice_padded.start)],
                mg,
                order = 1,
                mode = 'nearest',
                prefilter = True,
                # TODO add an output array here to reuse with paged dtoh copies.
            )
            if input_array_is_cupy:
                out[..., sl] += add_cp
            else:
                add_np = add_cp.get()
                outview = out[..., sl]
                ne.evaluate('add_np + outview', out=outview, casting='same_kind')

        # Free some RAM and VRAM by making it available for garbage collection
        del xc, yc, zc, mg, add_np, add_cp
        lowres_correction_array_sliced_cp = None
        lowres_correction_array_sliced_np = None

    if auto_dc_adjust:
        cp.cuda.get_current_stream().synchronize() # Prevent log messages from being printed prematurely
        if log_prefix is not None:
            logger.debug(log_prefix + f"Applying auto-adjusted DC offset...")
        numels = len(topzmask[0]) + len(botzmask[0])
        # approx_num_layers_zeros = numels / out.shape[0] / out.shape[1]
        if numels == 0:
            logger.error(
                ("" if log_prefix is None else log_prefix) +
                f"There were no zero-valued backprojections in either the top or bottom Z slice. The auto-DC adjustment cannot proceed. Skipping DC adjustment."
            )
        else:
            numels_thresh: int = 1000
            if (numels < numels_thresh): # Combined top and bottom layers
                logger.warning(
                    ("" if log_prefix is None else log_prefix) +
                    f"There are comparitively few (<{numels_thresh:d}) zero-valued backprojections in the top and bottom Z slices. The auto-DC adjustment may be incorrect."
                )
            zero_mean = np.concatenate([
                out[..., -1][topzmask].flatten(),
                out[...,  0][botzmask].flatten(),
            ]).mean()
            # zextremities_mean = (out[..., -1].mean() + out[...,  0].mean())/2
            if log_prefix is not None:
                logger.debug(log_prefix + f"Number of should-be zeros reference by DC correction: {numels:d}")
                logger.debug(log_prefix + f"Mean of should-be zeros referenced by DC correction: {zero_mean:.2g}")
                # logger.debug(log_prefix + f"Mean of top and bottom Z slice before DC correction: {zextremities_mean:.2g}")
            if tp is cp:
                cp.subtract(out, zero_mean, out=out)
            else:
                ne.evaluate('out - zero_mean', out=out, casting='same_kind')
            if log_prefix is not None:
                zero_mean = np.concatenate([
                    out[..., -1][topzmask].flatten(),
                    out[...,  0][botzmask].flatten(),
                ]).mean()
            #     zextremities_mean = (out[..., -1].mean() + out[...,  0].mean())/2
                logger.debug(log_prefix + f"Mean of should-be zeros referenced by DC correction, after DC correction: {zero_mean:.2g}")
            #     logger.debug(log_prefix + f"Mean of top and bottom Z slice after DC correction: {zextremities_mean:.2g}")

    if softener is not None:
        cp.cuda.get_current_stream().synchronize() # Prevent log messages from being printed prematurely
        if log_prefix is not None:
            logger.debug(log_prefix + f"Applying softener mask...")
        if tp is cp:
            out *= softener
        else:
            ne.evaluate('out * softener', out=out, casting='same_kind')

    # If requested, place the output in the output array
    if copy_to_array is not None:
        cp.cuda.get_current_stream().synchronize() # Prevent log messages from being printed prematurely
        if log_prefix is not None:
            logger.debug(log_prefix + f"Copying output data to the output array...")
        if type(copy_to_array) is np.ndarray and not input_array_is_cupy:
            lib.ne_tools.fastcopy(dst=copy_to_array, src=out)
        elif type(copy_to_array) is np.ndarray and input_array_is_cupy:
            out.get(out=copy_to_array)
        elif type(copy_to_array) is cp.ndarray and not input_array_is_cupy:
            copy_to_array.set(out)
        elif type(copy_to_array) is cp.ndarray and input_array_is_cupy:
            cp.copyto(dst=copy_to_array, src=out)
        else:
            raise ValueError(f"Output array is not an np.ndarray or a cp.ndarray.")

    if log_prefix is not None:
        logger.debug(log_prefix + f"Done applying cone-beam ramp filter.")

    if copy_to_array is not None:
        out = None
        if input_array_is_cupy:
            lib.gpu_tools.clear_cache()
        return copy_to_array
    else:
        return out
