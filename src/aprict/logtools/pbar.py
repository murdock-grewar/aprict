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
import typing

import rich.progress

from ..lib import gpu_tools
from . import rich_custombar


class TextColumnWithCallable(rich.progress.TextColumn):
    """A column acting like a TextColumn, but calling a callable each time it is rendered. Useful for updating information before rendering."""

    _call_on_refresh: typing.Callable[[], None]

    def __init__(
        self,
        *args,  # ruff: ignore[missing-type-args]
        call_on_refresh: typing.Callable[[], None],
        **kwargs,  # ruff: ignore[missing-type-kwargs]
    ) -> None:
        self._call_on_refresh = call_on_refresh
        super().__init__(*args, **kwargs)

    def render(self, *args, **kwargs):  # ruff: ignore[missing-return-type-undocumented-public-function, missing-type-args, missing-type-kwargs, undocumented-public-method]
        self._call_on_refresh()
        return super().render(*args, **kwargs)


class ProgressBar:
    """A progress bar which will print to stderr."""

    _cm: rich.progress.Progress
    _gpu_mon: gpu_tools.UtilisationAveragePercMon | None

    _maxval: float | int
    _desc: str
    _valunit: str
    _persist: bool

    _logger: logging.Logger
    _enable: bool
    _task: rich.progress.TaskID | None

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
        gpu_mon_interval_seconds: float = 0.2,
    ) -> None:
        self._task = None  # set this to None initially, so that the dynamically updated fields can see that the task is not yet present
        self._enable = enable
        self._desc = desc
        self._maxval = maxval
        self._gpu_mon = None
        if self._enable:
            self._logger = logging.getLogger() if logger is None else logger
            self._gpu_mon = (
                None
                if not show_gpu_usage
                else gpu_tools.UtilisationAveragePercMon(
                    gpu_indices=gpu_tools.all_gpu_indices(),
                    interval_seconds=gpu_mon_interval_seconds,
                )
            )
            refresh_per_second = (
                max(2 / gpu_mon_interval_seconds, 10) if show_gpu_usage else 10
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
                    TextColumnWithCallable(
                        call_on_refresh=self._refresh_gpu_perc,
                        text_format="GPUs {task.fields[gpu_perc_str]:s}",
                    ),
                )
            )
            self._cm = rich.progress.Progress(
                rich.progress.SpinnerColumn("dots"),
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
                refresh_per_second=refresh_per_second,
                speed_estimate_period=86400,  # period in seconds
            )

    def __enter__(self) -> ProgressBar:  # ruff: ignore[undocumented-magic-method]
        if self._enable:
            self._cm.__enter__()
            self._task = self._cm.add_task(
                self._desc, total=self._maxval, gpu_perc_str="---%"
            )
            self.update_to(0)
        return self

    def _refresh_gpu_perc(self) -> None:
        progress = self._cm
        assert (
            self._gpu_mon is not None
        )  # if we use this function, then the GPU mon obj should have been defined
        gpu_perc = self._gpu_mon.gpu_perc
        gpu_perc_str = (
            "---%" if gpu_perc is None else f"{round(gpu_perc): >3d}%"
        )
        if self._task is not None:
            progress.update(self._task, gpu_perc_str=gpu_perc_str)

    def advance(self, advance: float | int) -> None:
        """Advance the progress bar by a number of steps."""
        if self._enable and self._task is not None:
            return self._cm.advance(task_id=self._task, advance=advance)
        return None

    def update_to(self, val: float | int) -> None:
        """Update the progress bar value."""
        if self._enable and self._task is not None:
            progress: rich.progress.Progress = self._cm
            if self._gpu_mon is not None:
                gpu_perc = self._gpu_mon.gpu_perc
            else:
                gpu_perc = None
            gpu_perc_str = (
                "---%" if gpu_perc is None else f"{round(gpu_perc): >3d}%"
            )
            progress.update(
                self._task, completed=val, gpu_perc_str=gpu_perc_str
            )

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        if self._enable:
            self._cm.__exit__(exc_type, exc_val, exc_tb)
        if self._gpu_mon is not None:
            self._gpu_mon.close()
            self._gpu_mon = None
