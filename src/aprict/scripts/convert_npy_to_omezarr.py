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

"""Convert a numpy array on disk to an OME-Zarr-compliant .zarr directory."""

import pathlib
import types

import click
import click_extra as ce
import numpy as np

from ..lib import ome_zarr_tools

_progdesc = (
    "Convert a numpy array on disk to an OME-Zarr-compliant .zarr directory."
)


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
    "--ver",
    type=click.Choice(ome_zarr_tools.OMEZarrVersion),
    default=ome_zarr_tools.OMEZarrVersion.V0_5,
    help="The version of the OME-Zarr format.",
)
@ce.option(
    "--input",
    "-i",
    required=True,
    type=click.Path(
        path_type=pathlib.Path,
        allow_dash=False,
    ),
)
@ce.option(
    "--output",
    "-o",
    required=True,
    # Note that setting file_okay=False and dir_okay=False causes it to only accept uncreated paths
    type=click.Path(
        path_type=pathlib.Path,
        writable=True,
        file_okay=False,
        dir_okay=False,
        allow_dash=False,
    ),
)
@ce.option(
    "--voxwid-mm",
    required=True,
    type=click.FLOAT,
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[undocumented-public-function]
    ome_zarr_tools.nparr_to_zarr(
        arr=np.lib.format.open_memmap(args.input, mode="r"),
        out_zarr_path=args.output,
        omez_version=args.ver,
        voxwid_mm=args.voxwid_mm,
    )
