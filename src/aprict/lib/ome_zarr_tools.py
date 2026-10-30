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

import enum
from typing import TYPE_CHECKING

import dask.array
import numpy as np
import ome_zarr
import zarr

if TYPE_CHECKING:
    import pathlib


class OMEZarrVersion(enum.Enum):
    """Version of OME-Zarr to use."""

    V0_5 = "0.5"
    V0_6 = "0.6"


def nparr_to_zarr(
    arr: np.ndarray | np.memmap,
    *,
    voxwid_mm: float,
    out_zarr_path: pathlib.Path,
    omez_version: OMEZarrVersion = OMEZarrVersion.V0_5,
) -> None:
    """Convert a numpy array to OME-Zarr format."""
    # Validate args
    if out_zarr_path.is_dir():
        msg = f"Directory already exists: {out_zarr_path}"
        raise FileExistsError(msg)

    # Currently, neuroglancer only supports up to OME-Zarr 0.5
    verstr: str = omez_version.value

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


def zarr_to_nparr(
    zarr_path: pathlib.Path,
    *,
    out_arr_path: pathlib.Path,
) -> None:
    """Convert an OME-Zarr directory to a numpy array."""
    # Validate args
    if out_arr_path.is_file():
        msg = f"File already exists: {out_arr_path}"
        raise FileExistsError(msg)

    zarr_group = zarr.open_group(zarr_path, mode="r")
    ngff_image = ome_zarr.OMEZarrMultiscale.from_ome_zarr(zarr_group)
    dask_array = ngff_image.images[0].data
    arr = np.lib.format.open_memmap(
        out_arr_path, mode="w+", dtype=dask_array.dtype, shape=dask_array.shape
    )
    dask.array.store(dask_array, arr)  # ty: ignore[invalid-argument-type]
    arr.flush()
