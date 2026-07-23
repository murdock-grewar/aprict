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

"""Ramp kernels."""

import numpy as np


# Integrate[Exp[i w n L] Sinc [w L / (2 pi) ] Abs[w], {w, -pi/L, pi/L}], then take L = physpixsize
# Also obtainable as a regularised form of the non-band-limited integral with this Mathematica code:
# Simplify[Limit[Integrate[
#    Abs[w] Cos[w n L] Exp[-epsilon Abs[w n] L] Sinc[(w L)/2]
#    , {w, -Infinity, Infinity},
#    Assumptions -> L > 0 && epsilon > 0 && n > 0], epsilon -> 0]]
def ramp_kernel(
    n: int,  # Size of window. Must be odd.
    physpixsize: float | None = None,
) -> np.ndarray:
    """Produce a 1D kernel which is band-limited to a maximum frequency of `(1/2 * 1/physpixsize)` Hz, and operates on a 1D array with pixel edge length `physpixsize` by including a sinc(f) correction (with f measured in cycles per pixel)."""
    if n % 2 == 0:
        msg = "Window length must be odd."
        raise ValueError(msg)
    halfn = (n + 1) // 2
    arange: np.ndarray = (
        np.arange(2 * halfn - 1, dtype=np.float64) - halfn + 1
    )  # NOTE: np.float32 is not precise enough! Use np.float64
    return 8 / (1 - 4 * np.square(arange)) / (physpixsize or 1)


# Integrate[Exp[i w n L] Abs[w], {w, -pi/L, pi/L}], then take L = physpixsize
def ramp_kernel_ramlak(
    n: int,  # Size of window. Must be odd.
    physpixsize: float | None = None,
) -> np.ndarray:
    """Produce a 1D kernel which is band-limited to a maximum frequency of `(1/2 * 1/physpixsize)` Hz. However, the Ram-Lak filter does NOT account for the discreteness of the kernel by adding the sinc(f) factor (with f measured in cycles per pixel)."""
    if n % 2 == 0:
        msg = "Window length must be odd."
        raise ValueError(msg)
    halfn = (n + 1) // 2
    arange: np.ndarray = (
        np.arange(2 * halfn - 1, dtype=np.float64) - halfn + 1
    )  # NOTE: np.float32 is not good enough for high-precision reconstruction! It results in tiny DC error! Using np.float64 here fixes it!
    return (
        np.square(np.pi)
        * (2 * np.sinc(arange) - np.square(np.sinc(1 / 2 * arange)))
        / (physpixsize or 1)
    )


def hilbert_transform_kernel_approx(
    n: int,  # Size of window. Must be odd.
    physpixsize: float | None = None,
) -> np.ndarray:
    """Produce a 1D kernel which implements the Hilbert transform when linearly convolved."""
    if n % 2 == 0:
        msg = "Window length must be odd."
        raise ValueError(msg)
    halfn = (n + 1) // 2
    arange: np.ndarray = (
        np.arange(2 * halfn - 1, dtype=np.float64) - halfn + 1
    )
    zmask: np.ndarray = np.mod((
        np.arange(2 * halfn - 1, dtype=np.int32) - halfn + 1
    ), 2) == 0
    kern = np.true_divide(2, np.pi * arange * (physpixsize or 1))
    kern[zmask] = 0
    return kern
