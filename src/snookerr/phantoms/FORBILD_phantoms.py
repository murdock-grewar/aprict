# noqa: N999
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

"""Provides FORBILD phantoms from the files published by the Arbeitsgruppe Phantome."""

import dataclasses
import pathlib

import numpy as np
import regex
import scipy

from .. import data
from ..lib import process_pha_file as ppf
from ..volume import FBVolume

path_to_FORBILD_src = pathlib.Path(str(data.__path__[0])).joinpath("FORBILD")  # noqa: N816


# ruff: disable[N802,D400]


class _MetaPhantomContainer(type):
    @property
    def HeadPhantom(cls) -> FBVolume:
        """HeadPhantom.pha"""
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("HeadPhantom.pha")
            )
        )
        # Phantom is defined with centimeters as the unit of length.
        # Convert it to mm here.
        fbv.scale_len(10)
        return fbv

    @property
    def LungPhantom3Generations(cls) -> FBVolume:
        """LungPhantom3Generations.pha"""
        # DEV NOTE: make sure this phantom loads the reference file published by the Arbeitsgruppe Phantome. This is used in a unittest to ensure that the generation procedure for N generations matches in this case.
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("LungPhantom3Generations.pha")
            )
        )
        # Phantom is defined upside down. Fix it here.
        fbv.flip(axis="z")
        return fbv


class PhantomContainer(metaclass=_MetaPhantomContainer):
    """A container for the phantom objects."""

    @classmethod
    def _LungPhantom_phastring(cls, generation: int) -> str:  # noqa: C901
        """Generate the Lung phantom (by Marc Kachelrieß) up to generation n."""
        if type(generation) is not int or generation < 2:
            msg = "'generation' must be an integer >=2"
            raise ValueError(msg)
        # Lindenmayer System
        s = "T[L[s][b]][R[s][b]]"
        for _ in range(generation - 2):
            s = s.replace("b", "B[X][Y]").replace("s", "S[X][Y]")
            s = s.replace("X", "b").replace("Y", "s")
        s = s.replace("b", "B").replace("s", "S")
        # Construct a dictionary tree of the terminal symbols T, L, R, B, S
        patt = regex.compile(r"(?V1)\[((?:[^\]\[]+|(?R))*+)\]")

        @dataclasses.dataclass(frozen=True)
        class Branch:
            generation: int
            scale: float
            length: float
            radius_outer: float
            radius_inner: float
            u: tuple[float, float, float]  # start point
            v: tuple[float, float, float]  # end point
            x: tuple[
                float, float, float
            ]  # defines the branch plane together with z
            y: tuple[
                float, float, float
            ]  # new branches will be rotated about y
            z: tuple[float, float, float]  # direction of branch

        def generate_branch(symbol: str, parent: Branch | None) -> Branch:
            if parent is None:
                if symbol not in ("T", "L", "R"):
                    raise ValueError()
            match symbol:
                case "T":
                    return Branch(
                        0,
                        1.0,
                        100.0,
                        9.0,
                        7.9,
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 100.0),
                        (1.0, 0.0, 0.0),
                        (0.0, 1.0, 0.0),
                        (0.0, 0.0, 1.0),
                    )
                case "L":
                    return Branch(
                        1,
                        0.86,
                        34.4,
                        5.59,
                        4.644,
                        (0.0, 0.0, 100.0),
                        (28.1788, 0.0, 119.731),
                        (0.573576, 0.0, -0.819152),
                        (0.0, 1.0, 0.0),
                        (0.819152, 0.0, 0.573576),
                    )
                case "R":
                    return Branch(
                        1,
                        0.86,
                        43.0,
                        6.45,
                        5.504,
                        (0.0, 0.0, 100.0),
                        (-35.2235, 0.0, 124.664),
                        (-0.573576, 1.22461e-16, -0.819152),
                        (-7.02405e-17, -1.0, -1.00314e-16),
                        (-0.819152, 0.0, 0.573576),
                    )
                case "B":
                    assert parent is not None
                    r = scipy.spatial.transform.Rotation.from_rotvec(
                        np.array(parent.y) * -25, degrees=True
                    )
                    #
                    generation = parent.generation + 1
                    scale = parent.scale * 0.86
                    length = scale * 50
                    radius_outer = scale * 8.5
                    radius_inner = scale * 7.4
                    u = parent.v
                    z = tuple(r.apply(parent.z))
                    x = tuple(
                        r.apply(parent.x)
                    )  # the web page isn't clear about how the new x should be determined
                    r2 = scipy.spatial.transform.Rotation.from_rotvec(
                        np.array(z) * 90,
                        degrees=True,  # I thought the angle here should have a minus sign, but no, it matches the 3rd generation example without it.
                    )
                    x = tuple(r2.apply(x))
                    v = tuple(np.array(u) + length * np.array(z))
                    y = tuple(np.cross(z, x))
                    return Branch(
                        generation,
                        scale,
                        length,
                        radius_outer,
                        radius_inner,
                        u,
                        v,
                        x,
                        y,
                        z,
                    )
                case "S":
                    assert parent is not None
                    r = scipy.spatial.transform.Rotation.from_rotvec(
                        np.array(parent.y) * 45, degrees=True
                    )
                    #
                    generation = parent.generation + 1
                    scale = parent.scale * 0.86
                    length = scale * 40
                    radius_outer = scale * 7.0
                    radius_inner = scale * 5.9
                    u = parent.v
                    z = tuple(r.apply(parent.z))
                    x = tuple(
                        r.apply(parent.x)
                    )  # the web page isn't clear about how the new x should be determined
                    r2 = scipy.spatial.transform.Rotation.from_rotvec(
                        np.array(z) * 10,
                        degrees=True,  # I thought the angle here should have a minus sign, but no, it matches the 3rd generation example without it.
                    )
                    x = tuple(r2.apply(x))
                    v = tuple(np.array(u) + length * np.array(z))
                    y = tuple(np.cross(z, x))
                    return Branch(
                        generation,
                        scale,
                        length,
                        radius_outer,
                        radius_inner,
                        u,
                        v,
                        x,
                        y,
                        z,
                    )
                case _:
                    raise ValueError()

        def peel_layer(s: str, parent: Branch | None) -> dict:
            head = s[0]
            head_branch = generate_branch(symbol=head, parent=parent)
            blist = patt.findall(s)  # should either find 0 or 2
            if len(blist) == 0:
                return {head_branch: {}}
            b1, b2 = blist
            d = {head_branch: [b1, b2]}
            for ind, b in enumerate(d[head_branch]):
                d[head_branch][ind] = peel_layer(b, head_branch)
            return d

        d = peel_layer(s, None)
        # Flatten the dictionary into a list of objects
        # No two objects should be the same, so it should be OK to continue to use them as keys...
        lst: list[Branch] = []
        while len(d.keys()) > 0:
            keys = list(d.keys())
            for key in keys:
                lst.append(key)
                for val in d[key]:
                    d.update(val)  # unpack branches into root layer of dict
                del d[key]  # delete the old branch
        # First, print out the outer tubes and spheres
        ans = ""  # we will build the final string from this
        ans += "#define OUTERRHO 1.0\n"
        ans += "#define INNERRHO 0.1\n"
        ans += "#define RELEVANCE relevance\n"
        ans += "\n"
        ans += "// Outer Objects:\n"
        ans += "\n"
        gen: int = -1  # keep track of generation. When it increments, print a comment line saying so
        for b in lst:
            if b.generation > gen:
                gen = b.generation
                ans += "\n"
                ans += f"// Generation {gen:02d}\n"
            pos = (np.array(b.u) + np.array(b.v)) / 2
            ans += (
                "{"
                + f"[Cylinder: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r={b.radius_outer}mm axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=OUTERRHO RELEVANCE"
                + "}\n"
            )
            ans += (
                "{"
                + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_outer}mm] rho=OUTERRHO RELEVANCE"
                + "}\n"
            )
            ans += "\n"
        # Second, print out the inner tubes and spheres
        ans += "\n"
        ans += "// Inner Objects:\n"
        ans += "\n"
        gen: int = -1  # keep track of generation. When it increments, print a comment line saying so
        for b in lst:
            if b.generation > gen:
                gen = b.generation
                ans += "\n"
                ans += f"// Generation {gen:02d}\n"
            pos = (np.array(b.u) + np.array(b.v)) / 2
            ans += (
                "{"
                + f"[Cylinder: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r={b.radius_inner}mm axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=INNERRHO RELEVANCE"
                + "}\n"
            )
            ans += (
                "{"
                + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_inner}mm] rho=INNERRHO RELEVANCE"
                + "}\n"
            )
        return ans

    @classmethod
    def LungPhantom(cls, generation: int) -> FBVolume:
        """Generate the Lung phantom (by Marc Kachelrieß) up to generation n."""
        phastr: str = cls._LungPhantom_phastring(generation)
        # Create the shapes and return the constucted volume
        fbv = FBVolume(ppf.convert_ph_str_to_shapes(phastr))
        # Phantom is defined upside down. Fix it here.
        fbv.flip(axis="z")
        return fbv


# ruff: enable[N802,D400]
