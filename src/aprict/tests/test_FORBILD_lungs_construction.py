# ruff: ignore[invalid-module-name]
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

"""Tests for the generation of the FORBILD lungs phantom."""

import numpy as np

from .. import (  # import directly from root aprict namespace
    FBVolume,
    phantoms,
)


def test_correctness_of_Lungs_3Generations() -> None:  # ruff: ignore[invalid-function-name]
    """Test that the 3rd generation lung phantom matches the definition published by the Arbeitsgruppe Phantome."""
    lungs3: FBVolume = phantoms.FORBILD.LungPhantom(generation=3)
    lungs3_ref: FBVolume = phantoms.FORBILD.LungPhantom3Generations()

    assert len(lungs3.shapes) == len(lungs3_ref.shapes), (
        "The generated and reference lungs phantoms do not have the same number of primitive shapes."
    )

    def round_shape(s_in, digits: int = 2):  # ruff: ignore[missing-type-function-argument, missing-return-type-private-function]
        s = s_in.copy()
        scalar_keys = {
            "rho",
            "x",
            "y",
            "z",
            "dx",
            "dy",
            "dz",
            "r",
            "r1",
            "r2",
            "l",
        }
        axis_keys = {"cas1", "cas2", "cas3", "a_x", "a_y", "a_z", "axis"}
        for key in scalar_keys.intersection(s.dtype.names):
            s[key] = np.round(s[key], digits)
        for key in axis_keys.intersection(s.dtype.names):
            for subkey in scalar_keys.intersection(s[key].dtype.names):
                s[key][subkey] = np.round(s[key][subkey], digits)
        return s

    lungs3_shapes = [round_shape(s) for s in lungs3.shapes]
    lungs3_ref_shapes = [round_shape(s) for s in lungs3_ref.shapes]
    for shape in lungs3_ref_shapes:
        assert shape in lungs3_shapes, (
            "A shape inside the reference lungs phantom was not found in the generated lungs phantom."
        )  # note that 'in' syntax also checks for equality
