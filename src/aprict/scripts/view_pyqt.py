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

"""Script to display a 3D numpy array on disk using pyqtgraph."""

import pathlib
import types

import click
import click_extra as ce

# import cloup
import numpy as np

from aprict import image_view as imview
from aprict import lib

_progdesc = "Launch a pyqtgraph window to view a 3D numpy array on-disk."


@ce.command(
    params=None,
    help=_progdesc,
    context_settings={
        "show_default": True,
        "show_choices": True,
        "terminal_width": None,
        "max_content_width": 300,
    },
)
@ce.option(
    "--plotmin",
    "--pmin",
    type=lib.click_tools.MaybeType(click.FLOAT),
    default=None,
)
@ce.option(
    "--plotmax",
    "--pmax",
    type=lib.click_tools.MaybeType(click.FLOAT),
    default=None,
)
@ce.argument(
    "npy",
    type=click.Path(
        path_type=pathlib.Path,
        writable=False,  # no need for write permissions
        file_okay=True,  # accept a file
        dir_okay=False,  # don't accept a directory
    ),
    metavar="*.npy",
    help="Path to a numpy array file.",
)
@ce.option(
    "--memmap",
    type=lib.click_tools.BoolArgType(),
    default=False,
    help="Whether to memory-map the numpy array instead of loading it into RAM.",
)
@ce.option(
    "--transpose",
    type=lib.click_tools.IntTupleArgTypeFactory(numints=3),
    default=(0, 1, 2),
    help="The order in which to transpose axes of the volume.",
)
@ce.option(
    "--title",
    type=lib.click_tools.MaybeType(click.STRING),
    default=None,
    help="The string to display in the title of the window.",
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def validate_args(args: types.SimpleNamespace) -> None:
    """Validate a set of args."""
    # Check that neuroglancer is available
    if not imview._pyqtgraph_installed:  # ruff: ignore[private-member-access]
        msg = "pyqtgraph not able to be imported. (Did you install the `gui` extra?)"
        raise ModuleNotFoundError(msg)

    axes = args.transpose
    if len(axes) != 3 or 0 not in axes or 1 not in axes or 2 not in axes:
        msg = "Malformed --transpose argument."
        raise ce.UsageError(msg)


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[undocumented-public-function]
    validate_args(args)

    arr = (
        np.lib.format.open_memmap(args.npy)
        if args.memmap
        else np.load(args.npy)
    )
    arr = arr.transpose(args.transpose)
    imview.plot_volume(
        arr, plotmin=args.plotmin, plotmax=args.plotmax, title=args.title
    )


if __name__ == "__main__":
    main()
