# image_view.py
# Copyright (C) 2024-2026  Murdock Grewar
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
"""Helper methods to simply plot some projection/volume data."""

import logging
import pathlib
from typing import TYPE_CHECKING

import cupy as cp
import imageio.v3
import numpy as np

if TYPE_CHECKING:
    from pyqtgraph.Qt.QtWidgets import QApplication, QMainWindow

_pyqtgraph_installed: bool
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets

    _pyqtgraph_installed = True
except ImportError:
    _pyqtgraph_installed = False
    # logging.warning("Could not import 'pytqgraph' module.")


def plot_projections(
    arr: np.ndarray,  # shape: (num_radiographs, rad_width, rad_height)
    **kwargs,  # ruff: ignore[missing-type-kwargs]
) -> None:
    """Plot a projection array shaped like (numprojs, det_xwid, det_ywid)."""
    _plot_attenuation(arr.transpose((1, 2, 0)), **kwargs)


def plot_volume(
    arr: np.ndarray,
    **kwargs,  # ruff: ignore[missing-type-kwargs]
) -> None:
    """Plot a volume array shaped like (X,Y,Z)."""
    _plot_attenuation(arr, **kwargs)


def _plot_attenuation(
    # data: Union[np.ndarray,cp.ndarray],
    data: np.ndarray,
    init_zslicenum: int | None = None,
    title: str | None = None,
    show: bool = True,
    plotmin: float | None = None,
    plotmax: float | None = None,
) -> tuple[QApplication, QMainWindow, pg.ImageView] | None:

    if not _pyqtgraph_installed:
        logging.warning(
            "Failed to show a pyqtgraph plot because the 'pyqtgraph' module could not be imported. (Is it installed?)"
        )
        return None
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder="row-major")

    app = pg.mkQApp("Attenuation Plot")

    # Create window with ImageView widget
    win = QtWidgets.QMainWindow()
    win.resize(1200, 800)
    imv = pg.ImageView(discreteTimeLine=True)
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle("pyqtgraph plot" if title is None else title)
    imv.setHistogramLabel("Attenuation")

    if type(data) is np.ndarray:
        dnp = data.view()
    else:
        dnp = cp.asnumpy(data).view()

    # Need to transpose/flip to be plotted correctly.
    # The way that `imv.setImage(array)` behaves:
    # - axis 0 is the timeslice
    # - axis 1 is the vertical axis. The lowest index is at the top of the image.
    # - axis 2 is the horizontal axis. The lowest index is at the left of the image.
    # The way the input array is passed in:
    # - axis 0 is the x coordinate. The lowest index is the least x.
    # - axis 1 is the y coordinate. The lowest index is the least y.
    # - axis 2 is (for a volume) the z coordinate. The lowest index is the least z.
    dnp = dnp.transpose((2, 1, 0))
    dnp = np.flip(dnp, axis=1)
    imv.setImage(dnp)
    imv.setLevels(
        min=(np.amin(dnp) if plotmin is None else plotmin),
        max=(np.amax(dnp) if plotmax is None else plotmax),
    )
    # imv.play(10)
    imv.setCurrentIndex(
        dnp.shape[0] // 2 if init_zslicenum is None else init_zslicenum
    )
    # imv.autoLevels()
    # imv.autoRange()

    # Set a custom color map
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    # imv.setColorMap(cmap)

    # Start up with an ROI
    imv.ui.roiBtn.setChecked(False)
    imv.roiClicked()

    if show:
        pg.exec()
    return app, win, imv


def save_volslice(
    path: pathlib.Path,
    imarr: np.ndarray | cp.ndarray,
    axis: int,
    index_of_slice: int | None = None,
    pmin: float | None = None,
    pmax: float | None = None,
) -> None:
    """Save a 2D slice of a volume as a .png."""
    index_of_slice = (
        imarr.shape[axis] // 2 if index_of_slice is None else index_of_slice
    )
    volslice = imarr[(slice(None),) * axis + (index_of_slice,)]
    save_2d_arr(path, volslice, pmin=pmin, pmax=pmax)


def save_2d_arr(
    path: pathlib.Path,
    imarr: np.ndarray | cp.ndarray | pathlib.Path | str,
    pmin: float | None = None,
    pmax: float | None = None,
) -> None:
    """Save a 2D array as a .png."""
    if type(imarr) in (str,) or issubclass(type(imarr), pathlib.PurePath):
        imarr = np.load(imarr)  # ty:ignore[invalid-argument-type]
    if type(imarr) not in (np.ndarray, cp.ndarray, np.memmap):
        msg = "Unsupported type for 'imarr'."
        raise ValueError(msg)
    imarr_np = imarr.get() if type(imarr) is cp.ndarray else imarr  # ty:ignore[unresolved-attribute]
    volslice: np.ndarray = np.flip(imarr_np.transpose(), axis=0).copy()  # ty:ignore[unresolved-attribute]
    if pmin is None:
        pmin = volslice.min()
    if pmax is None:
        pmax = volslice.max()
    volslice = ((volslice - pmin) / (pmax - pmin)) * 255
    volslice = np.maximum(np.uint8(0), np.minimum(np.uint8(255), volslice))
    volslice = volslice.astype(np.uint8)
    imageio.v3.imwrite(
        path, volslice, mode="L"
    )  # , cmin=pmin, cmax=pmax, mode='L')
