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

# Note: this is not my code. I have copied it from
# user 'svioletg' who shared it on GitHub, from their GitHub Gist:
# https://gist.github.com/svioletg/621c5dc14e45b36b58ac576dceb9b75b
"""Classes for customizing the style of rich's progress bar column."""

from dataclasses import dataclass, fields
from types import NoneType, UnionType
from typing import TYPE_CHECKING, get_args, override

from rich.progress import BarColumn, Task
from rich.progress_bar import ProgressBar
from rich.segment import Segment

if TYPE_CHECKING:
    from rich.console import Console, ConsoleOptions, RenderResult
    from rich.style import Style, StyleType
    from rich.table import Column


@dataclass(frozen=True)
class BarChars:
    """Characters to use in custom progress bars."""

    edge_l: str | None = None
    """Placed at the left edge of the bar if truthy, otherwise uses ``empty`` or ``filled``."""
    edge_r: str | None = None
    """Placed at the right edge of the bar if truthy, otherwise uses ``empty`` or ``filled``."""
    empty: str = "━"
    tip: str | None = "╸"
    """Placed after the end of the filled bar if truthy, otherwise uses ``empty``."""
    filled: str = "━"

    def __post_init__(self) -> None:  # ruff: ignore[undocumented-magic-method]
        # This is just for validation; you can omit __post_init__ entirely if you don't care about checking
        for fld in fields(self.__class__):
            val = getattr(self, fld.name)
            if val is None:
                if (isinstance(fld.type, UnionType)) and (
                    NoneType in get_args(fld.type)
                ):
                    continue
                msg = f"{self.__class__.__name__}.{fld.name} expected type '{fld.type}', got None"
                raise TypeError(msg)
            if len(val) != 1:
                msg_0 = f"{self.__class__.__name__}.{fld.name} must be one character in length: {val!r}"
                raise ValueError(msg_0)


BAR_CHARS_DEFAULT = BarChars()
BAR_CHARS_BIGBLOCK = BarChars(filled="█", tip="▒", empty="░")
BAR_CHARS_DOCKER_STYLE = BarChars(
    edge_l="[", edge_r="]", empty=" ", tip=">", filled="="
)


class CustomProgressBar(ProgressBar):
    """A progress bar which can be customized by changing its :data:`bar_chars` value."""

    bar_chars: BarChars

    def __init__(
        self,
        *,
        bar_chars: BarChars | None = None,
        total: float | None = 100.0,
        completed: float = 0,
        width: int | None = None,
        pulse: bool = False,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        animation_time: float | None = None,
    ) -> None:
        self.bar_chars = bar_chars or BAR_CHARS_DEFAULT

        super().__init__(
            total=total,
            completed=completed,
            width=width,
            pulse=pulse,
            style=style,
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=pulse_style,
            animation_time=animation_time,
        )

    @override
    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        ascii_only: bool = options.legacy_windows or options.ascii_only
        bar_chars: BarChars = (
            BAR_CHARS_DEFAULT if ascii_only else self.bar_chars
        )
        width: int = min(self.width or options.max_width, options.max_width)
        bar_width: int = (
            width - bool(self.bar_chars.edge_l) - bool(self.bar_chars.edge_r)
        )

        should_pulse: bool = self.pulse or self.total is None
        if should_pulse:
            yield from self._render_pulse(console, width, ascii=ascii_only)
            return

        style: Style = console.get_style(self.style)
        completed: float = min(max(0, self.completed), self.total)
        is_finished: bool = self.completed >= self.total

        complete_style: Style = console.get_style(
            self.finished_style if is_finished else self.complete_style
        )

        bar_count = int(bar_width * (completed / self.total))
        remaining_bars = bar_width - bar_count - bool(bar_chars.tip)

        if bar_chars.edge_l:
            yield Segment(bar_chars.edge_l, complete_style)

        yield Segment(bar_chars.filled * bar_count, complete_style)
        if bar_chars.tip and (bar_count >= 0) and (remaining_bars >= 0):
            yield Segment(bar_chars.tip, complete_style)
        yield Segment(bar_chars.empty * remaining_bars, style)

        if bar_chars.edge_r:
            yield Segment(bar_chars.edge_r, complete_style)


class CustomBarColumn(BarColumn):
    """A progress bar column which can be customized by changing its :data:`bar_chars` value."""

    def __init__(
        self,
        *,
        bar_chars: BarChars | None = None,
        bar_width: int | None = 40,
        style: StyleType = "bar.back",
        complete_style: StyleType = "bar.complete",
        finished_style: StyleType = "bar.finished",
        pulse_style: StyleType = "bar.pulse",
        table_column: Column | None = None,
    ) -> None:
        self.bar_chars = bar_chars or BAR_CHARS_DEFAULT

        super().__init__(
            bar_width=bar_width,
            style=style,
            complete_style=complete_style,
            finished_style=finished_style,
            pulse_style=pulse_style,
            table_column=table_column,
        )

    @override
    def render(self, task: Task) -> CustomProgressBar:
        return CustomProgressBar(
            bar_chars=self.bar_chars,
            total=max(0, task.total) if task.total is not None else None,
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started,
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
        )
