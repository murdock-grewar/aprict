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

"""Module to provide a progress bar which prints to stderr."""

import logging

import rich.progress

from ..lib import gpu_tools
from . import rich_custombar


class ProgressBar:
    """A progress bar which will print to stderr."""

    _cm: rich.progress.Progress
    _gpu_mon: gpu_tools.UtilisationAveragePercMon

    _maxval: float | int
    _desc: str
    _valunit: str
    _persist: bool

    _logger: logging.Logger
    _enable: bool
    _task: rich.progress.TaskID

    def __init__(
        self,
        *,
        desc: str,
        logger: logging.Logger
        | None = None,  # This argument is needed to ensure that printing to the output goes through tqdm.write(...) before the logger. This ensures compatibility, so the progress bar is not reprinted when there are log lines in the middle of the progress bar progress. However, I'm not using tqdm at the moment, so it's not really necessary. any more...
        maxval: float | int,
        valunit: str | None = None,
        persist: bool = False,
        enable: bool = True,
        expand: bool = True,
        unit_is_bytes: bool = False,
        show_gpu_usage: bool = True,
    ) -> None:
        self._enable = enable
        self._desc = desc
        self._maxval = maxval
        if self._enable:
            self._logger = logging.getLogger() if logger is None else logger
            self._gpu_mon = gpu_tools.UtilisationAveragePercMon(
                gpu_indices=gpu_tools.all_gpu_indices(), interval_seconds=0.5
            )
            console = rich.console.Console(stderr=True)
            if unit_is_bytes:
                extracolumns = (
                    rich.progress.FileSizeColumn(),
                    rich.progress.TextColumn("/"),
                    rich.progress.TotalFileSizeColumn(),
                )
            else:
                extracolumns = (
                    ()
                    if valunit is None
                    else (
                        rich.progress.TextColumn(
                            text_format=valunit
                            + " #: {task.completed}/{task.total}"
                        ),
                    )
                )
            gpucolumns = (
                ()
                if not show_gpu_usage
                else (
                    rich.progress.TextColumn(
                        text_format="GPUs {task.fields[gpu_perc_str]:s}"
                    ),
                )
            )
            self._cm = rich.progress.Progress(
                rich.progress.TextColumn(
                    "[progress.description]{task.description}"
                ),
                rich_custombar.CustomBarColumn(
                    bar_chars=rich_custombar.BAR_CHARS_BIGBLOCK, bar_width=None
                ),
                rich.progress.TaskProgressColumn(),
                *extracolumns,
                rich.progress.TimeRemainingColumn(),
                ">",
                rich.progress.TimeElapsedColumn(),
                *gpucolumns,
                console=console,
                transient=not persist,
                expand=expand,
            )

    def __enter__(self) -> ProgressBar:  # ruff: ignore[undocumented-magic-method]
        if self._enable:
            self._cm.__enter__()
            self._task = self._cm.add_task(
                self._desc, total=self._maxval, gpu_perc_str="---%"
            )
            self.update_to(0)
        return self

    def update_to(self, val: float | int) -> None:
        """Update the progress bar value."""
        if self._enable:
            progress: rich.progress.Progress = self._cm
            gpu_perc = self._gpu_mon.gpu_perc
            gpu_perc_str = (
                "---%" if gpu_perc is None else f"{round(gpu_perc): >3d}%"
            )
            progress.update(
                self._task, completed=val, gpu_perc_str=gpu_perc_str
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        if self._enable:
            self._cm.__exit__(exc_type, exc_val, exc_tb)
