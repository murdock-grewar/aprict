# ImageView.py
# Copyright (C) 2024  Murdock Grewar
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

import pathlib
import typing
import imageio.v3
import logging
import numpy as np
import cupy as cp

from typing import *

_pyqtgraph_installed: bool
try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtWidgets
    _pyqtgraph_installed = True
except ImportError as e:
    _pyqtgraph_installed = False
    # logging.warning("Could not import 'pytqgraph' module.")

def plot_projections(
    arr: np.ndarray, # shape: (num_radiographs, rad_width, rad_height)
    **kwargs,
    ):
    plot_attenuation(arr.transpose((1,2,0)), **kwargs)

def plot_volume(
    arr: np.ndarray,
    **kwargs,
    ):
    plot_attenuation(arr, **kwargs)
def plot_attenuation(
      #data: Union[np.ndarray,cp.ndarray],
      data: np.ndarray,
      init_zslicenum: Optional[int] = None,
      title: Optional[str] = None,
      show: bool = True,
      plotmin: Optional[Any] = None,
      plotmax: Optional[Any] = None,
      ):
    if not _pyqtgraph_installed:
       logging.warning("Failed to show a pyqtgraph plot because the 'pyqtgraph' module could not be imported. (Is it installed?)")
       return
    # Interpret image data as row-major instead of col-major
    pg.setConfigOptions(imageAxisOrder='row-major')

    app = pg.mkQApp("Attenuation Plot")

    ## Create window with ImageView widget
    win = QtWidgets.QMainWindow()
    win.resize(1200, 800)
    imv = pg.ImageView(discreteTimeLine=True)
    win.setCentralWidget(imv)
    win.show()
    win.setWindowTitle('pyqtgraph plot' if title is None else title)
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
    dnp = dnp.transpose((2,1,0))
    dnp = np.flip(dnp, axis=1)
    imv.setImage(dnp)
    imv.setLevels(
      min = (np.amin(dnp) if plotmin is None else plotmin),
      max = (np.amax(dnp) if plotmax is None else plotmax),
    )
    #imv.play(10)
    imv.setCurrentIndex(dnp.shape[0]//2 if init_zslicenum is None else init_zslicenum)
    # imv.autoLevels()
    # imv.autoRange()

    ## Set a custom color map
    # cmap = pg.ColorMap(pos=np.linspace(0.0, 1.0, 6), color=colors)
    # imv.setColorMap(cmap)

    # Start up with an ROI
    imv.ui.roiBtn.setChecked(False)
    imv.roiClicked()

    if show:
      pg.exec()
    return app, win, imv

def save_volslice(path: pathlib.Path, imarr: np.ndarray | cp.ndarray, axis: int, index_of_slice: typing.Optional[int] = None, pmin: float = None, pmax: float = None):
    index_of_slice = imarr.shape[axis]//2 if index_of_slice is None else index_of_slice
    volslice = imarr[(slice(None),) * axis + (index_of_slice,)]
    save_2d_arr(path, volslice, pmin=pmin, pmax=pmax)

def save_2d_arr(path: pathlib.Path, imarr: np.ndarray | cp.ndarray | pathlib.Path, pmin: float = None, pmax: float = None):
    if type(imarr) in (str,) or issubclass(type(imarr), pathlib.PurePath):
        imarr = np.load(imarr)
    if type(imarr) not in (np.ndarray, cp.ndarray, np.memmap):
        raise ValueError("Unsupported type for 'imarr'.")
    imarr_np = imarr.get() if type(imarr) is cp.ndarray else imarr
    volslice: np.ndarray = np.flip(imarr_np.transpose(), axis=0).copy()
    if pmin is None:
        pmin = volslice.min()
    if pmax is None:
        pmax = volslice.max()
    volslice = ((volslice - pmin) / (pmax - pmin)) * 255
    volslice = np.maximum(np.uint8(0),np.minimum(np.uint8(255), volslice))
    volslice = volslice.astype(np.uint8)
    imageio.v3.imwrite(path, volslice, mode='L')#, cmin=pmin, cmax=pmax, mode='L')
