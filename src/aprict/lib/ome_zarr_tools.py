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

"""Tools to interface with the OME-Zarr format."""

from typing import TYPE_CHECKING

import ome_zarr

if TYPE_CHECKING:
    import pathlib

    import numpy as np


def nparr_to_zarr(
    arr: np.ndarray | np.memmap,
    *,
    voxwid_mm: float,
    out_zarr_path: pathlib.Path,
) -> None:
    """Convert a numpy array to OME-Zarr 0.5 format."""
    # Validate args
    if out_zarr_path.is_dir():
        msg = f"Directory already exists: {out_zarr_path}"
        raise FileExistsError(msg)

    # Currently, neuroglancer only supports up to OME-Zarr 0.5
    verstr: str = "0.5"

    image = ome_zarr.OMEZarrImage(
        data=arr,
        axes=["x", "y", "z"],
        scale={"x": voxwid_mm, "y": voxwid_mm, "z": voxwid_mm},
        axes_units={"x": "millimeter", "y": "millimeter", "z": "millimeter"},
    )
    multiscales = ome_zarr.OMEZarrMultiscale(
        image=image,
        scale_factors=[
            {"x": 2, "y": 2, "z": 2},
            {"x": 4, "y": 4, "z": 4},
            {"x": 8, "y": 8, "z": 8},
        ],
        method="local_mean",
        channel_names=["attenuation coefficient"],
    )
    multiscales.to_ome_zarr(out_zarr_path, version=verstr)
