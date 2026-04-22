# liveplot.py  # noqa: D100
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

import enum
import multiprocessing
import multiprocessing.connection
import sys
import typing

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.Qt.QtCore import QThread, pyqtSignal, pyqtSlot
from pyqtgraph.Qt.QtWidgets import QApplication, QGridLayout, QWidget


# ruff: disable[D101,D102,D103,ANN201,ANN204]
# fmt: off
class QLivePlotInstruction(enum.Enum):
    UPDATE_IMAGE    = 0
    AUTOLEVEL       = 1
    WIN_SET_VISIBLE = 2
# fmt: on


class QLivePlot(QWidget):
    imv: pg.ImageView

    def __init__(
        self,
        histogram_label: str | None = None,
        plotmin: float = 0,
        plotmax: float = 1,
    ) -> None:
        super().__init__()

        # Create ImageView widget from pyqtgraph
        self.imv = pg.ImageView(discreteTimeLine=True)
        # imv = pg.ImageView()
        self.imv.setHistogramLabel(
            "" if histogram_label is None else histogram_label
        )
        self.imv.setLevels(min=plotmin, max=plotmax)
        self.imv.setUpdatesEnabled(
            True
        )  # Don't know what this does... might not even be necessary

        # Set histogram autoranging behaviour so that the frequency axis adjusts automatically, but not the 'range' axis
        hist = self.imv.getHistogramWidget()
        hist.disableAutoHistogramRange()
        if hist.orientation == "vertical":
            hist.vb.enableAutoRange(hist.vb.XAxis, True)
        else:
            hist.vb.enableAutoRange(hist.vb.YAxis, True)

        # Make layout and put in ImageView widget
        layout = QGridLayout()
        layout.addWidget(self.imv)
        self.setLayout(layout)

    @pyqtSlot(QLivePlotInstruction, object)
    def do_action(
        self,
        instruction: QLivePlotInstruction,
        data: typing.Any,  # noqa: ANN401
    ) -> None:
        match instruction:
            case QLivePlotInstruction.UPDATE_IMAGE:
                # Setting the current index only seems to work when there is more than one layer...
                # It defaults to the 0th index. So, a hackish way to do this is only set the index
                # again if the index was above 0.
                ind = self.imv.currentIndex
                self.update(data)
                if ind > 0:
                    self.imv.setCurrentIndex(ind)
            case QLivePlotInstruction.AUTOLEVEL:
                self.imv.autoLevels()
            case _:
                pass

    def update(self, arr: np.ndarray):  # ty:ignore[invalid-method-override]
        self.imv.setImage(
            np.flip(arr, axis=1),
            autoRange=False,  # Pan and zoom
            autoLevels=False,  # Grey levels
            autoHistogramRange=False,  # This is necessary in combination with <HistogramWidget>.disableAutoHistogramRange()
        )


class QLivePlotWindow(QtWidgets.QMainWindow):
    qliveplot: QLivePlot

    def __init__(
        self, qliveplot: QLivePlot, title: str | None = None, show: bool = True
    ):
        super().__init__()
        self.qliveplot = qliveplot
        self.resize(1200, 800)
        self.setCentralWidget(qliveplot)
        self.setWindowTitle("Live plot" if title is None else title)
        show and self.show()

    @pyqtSlot(QLivePlotInstruction, object)
    def do_action(self, instruction: QLivePlotInstruction, data: typing.Any):  # noqa: ANN401
        match instruction:
            case QLivePlotInstruction.WIN_SET_VISIBLE:
                self.setVisible(data)
            case _:
                pass


class QLivePlotWorker(QThread):
    signal_qliveplot_do_action = pyqtSignal(QLivePlotInstruction, object)
    signal_qliveplotwin_do_action = pyqtSignal(QLivePlotInstruction, object)

    qliveplotwin: QLivePlotWindow
    qliveplot: QLivePlot
    conn_input: multiprocessing.connection.Connection

    def __init__(
        self,
        qliveplotwin: QLivePlotWindow,
        conn_input: multiprocessing.connection.Connection,
    ):
        super().__init__()
        # Initialise class members
        self.qliveplotwin = qliveplotwin
        self.qliveplot = self.qliveplotwin.qliveplot
        self.conn_input = conn_input
        # Initialise (connect) signals
        self.signal_qliveplot_do_action.connect(self.qliveplot.do_action)
        self.signal_qliveplotwin_do_action.connect(self.qliveplotwin.do_action)

    def triage_instruction(
        self, instruction: QLivePlotInstruction
    ) -> pyqtSignal:
        match instruction:
            case QLivePlotInstruction.WIN_SET_VISIBLE:
                return self.signal_qliveplotwin_do_action  # ty:ignore[invalid-return-type]
            case _:
                return self.signal_qliveplot_do_action  # ty:ignore[invalid-return-type]

    def run(self):
        while self.qliveplot:  # Loop while qliveplot object is not None?
            instruction: QLivePlotInstruction
            data: typing.Any
            instruction, data = self.conn_input.recv()
            sig = self.triage_instruction(instruction)
            sig.emit(instruction, data)  # ty:ignore[unresolved-attribute]


def make_dedicated_live_plot_window(
    conn_input: multiprocessing.connection.Connection,  # Connection object on which to call .recv() to get new data.
    title: str | None = None,
    histogram_label: str | None = None,
    plotmin: float | None = None,
    plotmax: float | None = None,
    show: bool = True,
):
    app = QApplication(sys.argv)
    qliveview = QLivePlot(histogram_label, plotmin, plotmax)  # ty:ignore[invalid-argument-type]
    qliveplotwin = QLivePlotWindow(qliveview, title, show)
    worker = QLivePlotWorker(qliveplotwin, conn_input)
    worker.start()
    sys.exit(app.exec())


class QLivePlotInstructionDispatcher:
    pipe_out: multiprocessing.connection.Connection

    def __init__(self, pipe_out: multiprocessing.connection.Connection):
        self.pipe_out = pipe_out

    def manual_instruction(
        self,
        instruction: QLivePlotInstruction,
        data: typing.Any,  # noqa: ANN401
    ):
        self.pipe_out.send((instruction, data))

    def update_image(self, arr: np.ndarray):
        self.manual_instruction(QLivePlotInstruction.UPDATE_IMAGE, arr)

    def set_visible(self, visible: bool):
        self.manual_instruction(QLivePlotInstruction.WIN_SET_VISIBLE, visible)


def start_live_plot_2d(
    title: str | None = None,
    histogram_label: str | None = None,
    plotmin: float | None = None,
    plotmax: float | None = None,
    show: bool = True,
) -> tuple[multiprocessing.Process, QLivePlotInstructionDispatcher]:
    (conn_parent, conn_child) = multiprocessing.Pipe(duplex=True)
    p = multiprocessing.Process(
        target=make_dedicated_live_plot_window,
        args=(conn_child, title, histogram_label, plotmin, plotmax, show),
    )
    instruction_dispatcher: QLivePlotInstructionDispatcher = (
        QLivePlotInstructionDispatcher(conn_parent)
    )
    p.start()
    # p.join() # Blocks until the thread is complete.
    return p, instruction_dispatcher


# ruff: enable[D101,D102,D103,ANN201,ANN204]
