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

"""Tools for use with the neuroglancer package."""

from typing import TYPE_CHECKING

from . import import_tools

neuroglancer = import_tools.import_neuroglancer()


if TYPE_CHECKING:
    import numpy as np


def _add_ndarray_layer(
    state,  # ruff: ignore[missing-type-function-argument]
    input_array: np.ndarray | np.memmap,
    voxel_size_mm: float | None = None,
    plotmin: float | None = None,
    plotmax: float | None = None,
) -> None:

    # data = np.cast[np.float32](input_array)
    data = input_array

    if voxel_size_mm is None:
        dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"], scales=[1, 1, 1]
        )
    else:
        dimensions = neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units=["mm", "mm", "mm"],
            scales=[voxel_size_mm, voxel_size_mm, voxel_size_mm],
        )
    shader = f"#uicontrol invlerp normalized(range=[{data.min():.3g}, {data.max():.3g}], clamp=true)"
    shader += """
void main() {
  emitGrayscale(normalized());
}
"""

    #     shader_example="""
    # void main() {
    #   emitRGB(vec3(toNormalized(getDataValue(0)),
    #                toNormalized(getDataValue(1)),
    #                toNormalized(getDataValue(2))));
    # }
    # """

    state.dimensions = dimensions

    state.layout = "xy"
    state.crossSectionOrientation = [1, 0, 0, 0]
    state.projectionOrientation = [
        1,
        0,
        0,
        0,
    ]  # rotate 180 degrees about the x axis. I do this to flip the y axis so it points upwards rather than downwards.

    local_vol = neuroglancer.LocalVolume(
        data=data,
        dimensions=dimensions,
        voxel_offset=(0, 0, 0),
        # voxel_offset=(0, 20, 30, 15),
        downsampling=None,  # disables downsampling
    )
    # Can flip axes etc here if desired. Fourth column is translation
    coord_transform = neuroglancer.CoordinateSpaceTransform(
        output_dimensions=dimensions,
        matrix=[
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
    )

    source = neuroglancer.LayerDataSource(
        url=local_vol, transform=coord_transform
    )

    pmin = float(plotmin) if plotmin is not None else float(input_array.min())
    pmax = float(plotmax) if plotmax is not None else float(input_array.max())
    state.layers.append(
        name="recon",
        # layer=local_vol,
        layer=neuroglancer.ImageLayer(
            source=source,
            shader_controls={"normalized": {"range": [pmin, pmax]}},
        ),
        shader=shader,
    )


class ArrayViewer3D:
    """A context manager to serve a numpy volume at a URL while active."""

    port: int
    array: np.ndarray | np.memmap
    voxwid_mm: float | None
    url: str

    _initial_plotmin: float | None
    _initial_plotmax: float | None
    _wait_for_input_before_exit_prompt: str | None

    def __init__(
        self,
        arr: np.ndarray | np.memmap,
        plotmin: float | None = None,
        plotmax: float | None = None,
        port: int = 8080,
        voxwid_mm: float | None = None,
        wait_for_input_before_exit_prompt: str | None = None,
    ) -> None:
        self.port = port
        self.array = arr
        self.voxwid_mm = voxwid_mm
        self._initial_plotmin = plotmin
        self._initial_plotmax = plotmax
        self._wait_for_input_before_exit_prompt = (
            wait_for_input_before_exit_prompt
        )

    def __enter__(self) -> ArrayViewer3D:  # ruff: ignore[undocumented-magic-method]
        port = str(self.port)
        neuroglancer.set_server_bind_address(
            bind_address="127.0.0.1", bind_port=port
        )
        # neuroglancer.cli.handle_server_arguments(args)  # could be useful in the future
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            _add_ndarray_layer(
                s,
                self.array,
                voxel_size_mm=self.voxwid_mm,
                plotmin=self._initial_plotmin,
                plotmax=self._initial_plotmax,
            )
        self.url = "http://localhost:" + port + "/v/" + viewer.token + "/"
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        if self._wait_for_input_before_exit_prompt is not None:
            while response := input(self._wait_for_input_before_exit_prompt):  # ruff: ignore[unused-variable]
                pass
        neuroglancer.stop()


class OMEZarrViewer3D:
    """A context manager to serve an OME-Zarr volume at a URL while active."""

    port: int
    zarr_url: str
    url: str

    _wait_for_input_before_exit_prompt: str | None

    def __init__(
        self,
        zarr_url: str,
        port: int = 8080,
        wait_for_input_before_exit_prompt: str | None = None,
    ) -> None:
        self.port = port
        self.zarr_url = zarr_url
        self._wait_for_input_before_exit_prompt = (
            wait_for_input_before_exit_prompt
        )

    def __enter__(self) -> OMEZarrViewer3D:  # ruff: ignore[undocumented-magic-method]
        port = str(self.port)
        neuroglancer.set_server_bind_address(
            bind_address="127.0.0.1", bind_port=port
        )
        # neuroglancer.cli.handle_server_arguments(args)  # could be useful in the future
        viewer = neuroglancer.Viewer()
        with viewer.txn() as s:
            s.layers["image"] = neuroglancer.ImageLayer(source=self.zarr_url)
        self.url = "http://localhost:" + port + "/v/" + viewer.token + "/"
        return self

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        if self._wait_for_input_before_exit_prompt is not None:
            while response := input(self._wait_for_input_before_exit_prompt):  # ruff: ignore[unused-variable]
                pass
        neuroglancer.stop()
