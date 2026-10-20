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

"""Script to display a 3D numpy array on disk in neuroglancer."""

import argparse
import pathlib

import neuroglancer.cli  # might be safe to import neuroglancer.cli directly, but not neuroglancer, due to it messing up logging. Use lib.import_tools.import_neuroglancer() for that.
import numpy as np

from aprict import lib

try:
    from aprict.lib import ng_tools

    _mod_avail_neuroglancer = True
except ModuleNotFoundError:
    _mod_avail_neuroglancer = False


def main() -> None:  # ruff: ignore[undocumented-public-function]
    ap = argparse.ArgumentParser()
    neuroglancer.cli.add_server_arguments(ap)
    # --My args--
    ap.add_argument(
        "input", metavar="path/to/numpy/array.npy", type=pathlib.Path
    )
    ap.add_argument(
        "--memmap", type=lib.argparse_tools.BoolArgType, default=False
    )
    ap.add_argument(
        "--voxel-size-mm",
        "--voxwid-mm",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
    )
    ap.add_argument(
        "--plotmin",
        "--pmin",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
    )
    ap.add_argument(
        "--plotmax",
        "--pmax",
        type=lib.argparse_tools.MaybeType(float),
        default=None,
    )
    ap.add_argument("--port", type=int, default=8080)
    # -----------
    args = ap.parse_args()

    if not _mod_avail_neuroglancer:
        msg = "This script cannot be used because neuroglancer could not be imported."
        raise ModuleNotFoundError(msg)

    if args.memmap:
        input_array = np.lib.format.open_memmap(args.input, mode="r")
    else:
        input_array = np.load(args.input)

    with ng_tools.ArrayViewer3D(
        input_array,
        port=args.port,
        voxwid_mm=args.voxel_size_mm,
        plotmin=args.plotmin,
        plotmax=args.plotmax,
    ) as av:
        print(av.url)
        # Wait for user input before quitting
        while (response := input("Enter 'N' to quit: ")).lower() != "n":  # ruff: ignore[unused-variable]
            pass


if __name__ == "__main__":
    main()
