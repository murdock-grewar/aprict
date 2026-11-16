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

"""Script to display a 3D array using neuroglancer."""

import pathlib

# import platform  # can be used to find domain name of this machine via platform.node()
import sys
import types
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from urllib.parse import quote

import click
import click_extra as ce
import cloup
import numpy as np

from aprict import lib

try:
    from aprict.lib import ng_tools

    _mod_avail_neuroglancer = True
except ModuleNotFoundError:
    _mod_avail_neuroglancer = False

_progdesc = (
    "Launch a neuroglancer instance to view a 3D volume by file path or URI."
)


# Used to host a .zarr dir
class _CORSHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()


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
    "--port",
    type=click.INT,
    default=8080,
    help="Port on which to host the neuroglancer page.",
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
@ce.option_group(
    "resource specifier",
    ce.option(
        "--npy",
        type=click.Path(
            path_type=pathlib.Path,
            writable=False,  # no need for write permissions
            file_okay=True,  # accept a file
            dir_okay=False,  # don't accept a directory
        ),
        metavar="*.npy",
        help="Path to a numpy array file.",
    ),
    ce.option(
        "--zarr",
        type=click.Path(
            path_type=pathlib.Path,
            writable=False,  # no need for write permissions
            file_okay=False,  # don't accept a file
            dir_okay=True,  # accept a zarr directory
        ),
        metavar="*.zarr/",
        help="Path to a zarr directory.",
    ),
    ce.option(
        "--uri",
        type=click.STRING,
        help="A literal URI string to give to the neuroglancer instance. Can be used to specify a URL to a zarr, e.g. 'zarr://https://....:8080'",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option_group(
    "--npy options",
    ce.option(
        "--voxwid",
        type=click.FLOAT,
        help="Voxel width of numpy array.",
    ),
    ce.option(
        "--unit-len",
        type=click.Choice(lib.ome_zarr_tools.UnitLen),
        help="Unit length used to interpret '--voxwid' and also the meaning of the attenuation coefficient of the input volume for the numpy array, which has units of inverse length.",
    ),
    ce.option(
        "--npy-memmap",
        type=lib.click_tools.BoolArgType(),
        default=False,
        help="Whether to memory-map the numpy array instead of loading it into RAM.",
    ),
)
@ce.option_group(
    "--zarr options",
    ce.option(
        "--zarr-serve-port",
        type=click.INT,
        help="Port on which to serve the zarr array.",
        default=8001,
    ),
    ce.option(
        "--zarr-serve-ip",
        type=click.STRING,
        default="127.0.0.1",
        help="The IP of the interface to bind to and serve the data. Use 0.0.0.0 to serve on all interfaces.",
    ),
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def validate_args(args: types.SimpleNamespace) -> None:
    """Validate a set of args."""
    # Check that neuroglancer is available
    if not _mod_avail_neuroglancer:
        msg = "neuroglancer not able to be imported. (Did you install the `gui` extra?)"
        raise ModuleNotFoundError(msg)

    # If specifying a numpy array, then we need certain options
    # Otherwise, those options should not be present.
    if args.npy is not None:
        if args.voxwid is None:
            msg = "--voxwid must be provided if --npy is provided."
            raise ce.UsageError(msg)
        if args.unit_len is None:
            msg = "--unit-len must be provided if --npy is provided."
            raise ce.UsageError(msg)
    else:
        if args.voxwid is not None:
            msg = "--voxwid cannot be provided unless --npy is provided."
            raise ce.UsageError(msg)
        if args.unit_len is not None:
            msg = "--unit-len cannot be provided unless --npy is provided."
            raise ce.UsageError(msg)


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[undocumented-public-function]
    validate_args(args)

    # hostname: str = platform.node()

    if args.zarr is not None:
        # serve a local zarr
        zarr_path = args.zarr.resolve()
        zarr_server_port: int = args.zarr_serve_port
        ip: str = args.zarr_serve_ip
        zarr_server = ThreadingHTTPServer(
            (ip, zarr_server_port),
            partial(_CORSHandler, directory=str(zarr_path.parent)),
        )
        Thread(target=zarr_server.serve_forever, daemon=True).start()
        # source_url = f"zarr://http://{ip:s}:{zarr_server_port:d}/{quote(zarr_path.name)}"
        source_url = f"zarr://http://localhost:{zarr_server_port:d}/{quote(zarr_path.name)}"

        with ng_tools.URIViewer3D(
            source_url,
            port=args.port,
            plotmin=args.plotmin,
            plotmax=args.plotmax,
        ) as av:
            print(av.url, file=sys.stderr)
            # Wait for user input before quitting
            while (
                response := input(
                    "Enter 'N' to quit: ",
                )
            ).lower() != "n":
                pass
    elif args.npy is not None:
        # serve a local numpy array
        if args.npy_memmap:
            input_array = np.lib.format.open_memmap(args.npy, mode="r")
        else:
            input_array = np.load(args.npy)

        with ng_tools.ArrayViewer3D(
            input_array,
            port=args.port,
            voxwid=args.voxwid,
            unit_len=args.unit_len,
            plotmin=args.plotmin,
            plotmax=args.plotmax,
        ) as av:
            print(av.url, file=sys.stderr)
            # Wait for user input before quitting
            while (response := input("Enter 'N' to quit: ")).lower() != "n":
                pass
    elif args.uri is not None:
        # serve a specified URI string
        with ng_tools.URIViewer3D(
            args.uri,
            port=args.port,
            plotmin=args.plotmin,
            plotmax=args.plotmax,
        ) as av:
            print(av.url, file=sys.stderr)
            # Wait for user input before quitting
            while (response := input("Enter 'N' to quit: ")).lower() != "n":  # ruff: ignore[unused-variable]
                pass

    else:
        # This should never happen.
        raise NotImplementedError()


if __name__ == "__main__":
    main()
