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
    def CustomComposite(cls) -> FBVolume:
        hi, ab, th, he, ja = [
            FBVolume(
                ppf.convert_ph_file_to_shapes(
                    path_to_FORBILD_src.joinpath("custom_composite/" + fn)
                )
            )
            for fn in (
                "part_01_hip.pha",
                "part_02_abdomen.pha",
                "part_03_thorax.pha",
                "part_04_head.pha",
                "part_05_jaw.pha",
            )
        ]

        # Create the lungs phantom which will sit on top of everything
        lu = PhantomContainer.LungPhantom(
            fix=True, generation=10, taper=True, scale_factor=0.60
        )
        lu.translate(axis="z", amount=150)
        lu.scale_len(
            1.8
        )  # initially too small to fit in the thorax lung pockets

        # Convert cm to mm.
        ja.scale_len(10)
        he.scale_len(10)
        th.scale_len(10)
        ab.scale_len(10)
        hi.scale_len(10)

        # The jaw is not located correctly with respect to the head. Correct it here.
        ja.translate(axis="y", amount=-20)
        ja.translate(axis="z", amount=-120)
        # Translate the head up and slightly forwards
        amount = 360
        he.translate(axis="z", amount=amount)
        ja.translate(axis="z", amount=amount)
        amount = 0  # Never mind. Don't translate it forwards.
        he.translate(axis="y", amount=amount)
        ja.translate(axis="y", amount=amount)
        # Translate abdomen and hip down
        amount = -200
        ab.translate(axis="z", amount=amount)
        hi.translate(axis="z", amount=amount)
        # Translate the hips down further
        amount = -300
        hi.translate(axis="z", amount=amount)

        # Remove the background shapes making up the thoracical/abdominal cavity. We'll place these objects first in the list after composing the full object, so that they remoain in the background.
        # cavity = FBVolume(th.shapes[[2]]).superimpose_onto(FBVolume(ab.shapes[[0]]))
        cavity = FBVolume(th.shapes[[2]])
        ab = FBVolume(ab.shapes[1:])
        th = FBVolume(np.concatenate((th.shapes[0:2], th.shapes[3:]), axis=0))
        # Extend the cavity to meet the hips
        cavity.shapes[0]["z"] -= 50
        cavity.shapes[0]["l"] += 100
        # Superimpose in order
        v = ja.superimpose_onto(lu)  # jaws on top of lungs is better
        v = v.superimpose_onto(he)
        v = v.superimpose_onto(th)
        v = v.superimpose_onto(ab)
        v = v.superimpose_onto(hi)
        v = v.superimpose_onto(cavity)
        return v  # noqa: RET504

    @property
    def JawPhantom(cls) -> FBVolume:
        """The JawPhantom.txt"""
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("jaw/JawPhantom.pha")
            )
        )
        # Phantom is defined with centimeters as the unit of length.
        # Convert it to mm here.
        fbv.scale_len(10)
        return fbv

    @property
    def HeadPhantom(cls) -> FBVolume:
        """HeadPhantom.pha"""
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("head/HeadPhantom.pha")
            )
        )
        # Phantom is defined with centimeters as the unit of length.
        # Convert it to mm here.
        fbv.scale_len(10)
        return fbv

    @property
    def HipPhantom(cls) -> FBVolume:
        """HipPhantom.pha"""
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("hip/HipPhantom.pha")
            )
        )
        # Phantom is defined with centimeters as the unit of length.
        # Convert it to mm here.
        fbv.scale_len(10)
        return fbv

    @property
    def AbdomenPhantom(cls) -> FBVolume:
        """Fixed AbdomenPhantom.txt"""
        # The AbdomenPhantom.pha file has a bug in it. Load a manually fixed version.
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath(
                    "abdomen/AbdomenPhantom_fixed.pha"
                )
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
                path_to_FORBILD_src.joinpath(
                    "lungs/LungPhantom3Generations.pha"
                )
            )
        )
        # Phantom is defined upside down. Fix it here.
        fbv.flip(axis="z")
        return fbv

    @property
    def ThoraxPhantom(cls) -> FBVolume:
        """ThoraxPhantom.pha"""
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("thorax/ThoraxPhantom.pha")
            )
        )
        # Phantom is defined with centimeters as the unit of length.
        # Convert it to mm here.
        fbv.scale_len(10)
        return fbv


class PhantomContainer(metaclass=_MetaPhantomContainer):
    """A container for the phantom objects."""

    @classmethod
    def _LungPhantom_phastring(  # noqa: C901
        cls,
        generation: int,
        scale_factor: float
        | None = None,  # Leave at None for default configuration
        taper: bool | None = None,  # Leave at None for default configuration
        fix: bool
        | None = None,  # Leave at None for default configuration. Turning this True will fix what appear to be errors in the original phantom. These errors are particularly noticeable with e.g. taper=True.
    ) -> str:
        """Generate the Lung phantom (by Marc Kachelrieß) up to generation n."""
        if type(generation) is not int or generation < 2:
            msg = "'generation' must be an integer >=2"
            raise ValueError(msg)
        if scale_factor is None:
            scale_factor = 0.86  # default value is 0.86
        if taper is None:
            taper = False  # default phantom does not taper. It uses cylinders, not truncated cones.
        if fix is None:
            fix = False
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
            taper_factor: float  # how much it shrinks by the end of the tube
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
                        1,  # do not taper the trachea
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
                        0.86 if taper else 1,
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
                        0.86 if taper else 1,
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
                    scale = parent.scale * scale_factor
                    length = scale * 50
                    # The amount to taper increase the radius toward the root
                    # depends on the radius of the parent.
                    # The parent can be the smaller of the two parents,
                    # in which case it needs to taper by less than the full
                    # inverse scale_factor
                    radius_outer = scale * 8.5 * (7.5 / 8.5 if fix else 1)
                    radius_inner = radius_outer * (7.4 / 8.5)
                    taper_factor = (
                        radius_outer / parent.radius_outer if taper else 1
                    )
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
                        taper_factor,
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
                    scale = parent.scale * scale_factor
                    length = scale * 40
                    radius_outer = scale * 7.0 * (7.5 / 8.5 if fix else 1)
                    radius_inner = radius_outer * (5.9 / 7.0)
                    taper_factor = (
                        radius_outer / parent.radius_outer if taper else 1
                    )
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
                        taper_factor,
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

        # HOTFIX: preprocessing the file with pcpp takes a long time. It's single-threaded.
        # To speed this up, I'll not use these macros. I'll just directly generate the files with the values replaced.
        # ans += "#define OUTERRHO 1.0\n"
        # ans += "#define INNERRHO 0.1\n"
        # ans += "#define RELEVANCE relevance\n"

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
            if b.taper_factor == 1:
                ans += (
                    "{"
                    + f"[Cylinder: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r={b.radius_outer}mm axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=1.0 relevance"
                    + "}\n"
                )
                ans += (
                    "{"
                    + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_outer}mm] rho=1.0 relevance"
                    + "}\n"
                )
            else:
                ans += (
                    "{"
                    + f"[Cone: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r1={b.radius_outer / b.taper_factor}mm r2={b.radius_outer} axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=1.0 relevance"
                    + "}\n"
                )
                ans += (
                    "{"
                    + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_outer}mm] rho=1.0 relevance"
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
            if b.taper_factor == 1:
                ans += (
                    "{"
                    + f"[Cylinder: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r={b.radius_inner}mm axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=0.1 relevance"
                    + "}\n"
                )
                ans += (
                    "{"
                    + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_inner}mm] rho=0.1 relevance"
                    + "}\n"
                )
            else:
                ans += (
                    "{"
                    + f"[Cone: x={pos[0]}mm y={pos[1]}mm z={pos[2]}mm l={b.length}mm r1={b.radius_inner / b.taper_factor}mm r2={b.radius_inner}mm axis({b.z[0]}mm, {b.z[1]}mm, {b.z[2]}mm)] rho=0.1 relevance"
                    + "}\n"
                )
                ans += (
                    "{"
                    + f"[Sphere: x={b.v[0]}mm y={b.v[1]}mm z={b.v[2]}mm r={b.radius_inner}mm] rho=0.1 relevance"
                    + "}\n"
                )
        return ans

    @classmethod
    def LungPhantom(
        cls,
        generation: int,
        scale_factor: float
        | None = None,  # Leave at None for default configuration
        taper: bool | None = None,  # Leave at None for default configuration
        fix: bool
        | None = None,  # Leave at None for default configuration. Turning this True will fix what appear to be errors in the original phantom. These errors are particularly noticeable with e.g. taper=True.
    ) -> FBVolume:
        """Generate the Lung phantom (by Marc Kachelrieß) up to generation n."""
        phastr: str = cls._LungPhantom_phastring(
            generation=generation,
            scale_factor=scale_factor,
            taper=taper,
            fix=fix,
        )
        # Create the shapes and return the constucted volume
        fbv = FBVolume(
            ppf.convert_ph_str_to_shapes(phastr, skip_preprocessing=True)
        )
        # Phantom is defined upside down. Fix it here.
        fbv.flip(axis="z")
        return fbv


# ruff: enable[N802,D400]
