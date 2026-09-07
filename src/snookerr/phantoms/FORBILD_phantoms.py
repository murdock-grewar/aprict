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

"""Provides FORBILD phantoms from the files published by the Arbeitsgruppe Phantome."""

import dataclasses
import itertools
import pathlib

import numpy as np
import regex
import scipy

from .. import data
from ..cuda.structs import FORBILD_volumes as fv
from ..lib import process_pha_file as ppf
from ..volume import FBVolume

path_to_FORBILD_src = pathlib.Path(str(data.__path__[0])).joinpath("FORBILD")  # ruff: ignore[mixed-case-variable-in-global-scope]


# ruff: disable[invalid-function-name,missing-trailing-period]


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
        return v  # ruff: ignore[unnecessary-assign]

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

    @property
    def RockCanisterAl(cls) -> FBVolume:
        fbv = FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath(
                    "custom_other/rock_canister_01.pha"
                )
            )
        )
        # Scale the attenuation coefficient so that the canister has a value of around 0.55 mm^{-1} for aluminium
        fbv.multiply_attenuation(0.55)
        return fbv

    @property
    def RockCanisterCavities1e5(cls) -> FBVolume:
        return FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath(
                    "custom_other/rock_canister_01_cavities_1e5.pha"
                )
            )
        )

    @property
    def UnitCyl_z(cls) -> FBVolume:
        return FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("custom_other/unit_Cyl_z.pha")
            )
        )


class PhantomContainer(metaclass=_MetaPhantomContainer):
    """A container for the phantom objects."""

    @classmethod
    def _LungPhantom_phastring(  # ruff: ignore[complex-structure]
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

    @classmethod
    def TallRockPhantom(  # ruff: ignore[complex-structure]
        cls,
        num_interior_objects: int = 10000,
        *,
        h: float,
        r: float,
        rng_seed: int | None = None,
        caps_height: float | None = None,
        middle_band_height: float | None = None,
        num_knurls: int | None = None,
        outer_container_rho: float | None = None,
        inner_container_rho: float | None = None,
    ) -> FBVolume:
        """Generate a pseudorandom tall rock phantom."""
        # Calling rng.random() will produce a float in [0,1)
        rng = np.random.default_rng(rng_seed)
        rng_knurl = np.random.default_rng((rng_seed or 0) + 1)

        shapes = []

        outer_container_rho = outer_container_rho or 1
        inner_container_rho = inner_container_rho or 0.05
        num_knurls = num_knurls or 0

        # Pseudorandom geometric parameters
        outer_radius = r * (4 / 5 + 1 / 5 * rng.random())
        inner_radius = outer_radius * (0.9 + 0.05 * rng.random())
        offset_angle = 2 * np.pi * rng.random()
        offset_radius = (r - outer_radius) * rng.random()
        offset_x = offset_radius * np.cos(offset_angle)
        offset_y = offset_radius * np.sin(offset_angle)

        # Make rock container -- it will be contained within the radius r,
        # but will be a little smaller and have randomised position
        cyl_outer = np.array(0, dtype=fv.Shape)
        cyl_outer["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
        cyl_outer["r"] = outer_radius
        cyl_outer["x"] = offset_x
        cyl_outer["y"] = offset_y
        cyl_outer["rho"] = outer_container_rho
        cyl_outer["l"] = h
        if num_knurls > 0:
            # In this case, the knurls will fill out the remaining radius
            cyl_outer["r"] = (outer_radius + inner_radius) / 2
        shapes.append(cyl_outer)
        cyl_inner = np.array(0, dtype=fv.Shape)
        cyl_inner["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
        cyl_inner["r"] = inner_radius
        cyl_inner["x"] = offset_x
        cyl_inner["y"] = offset_y
        cyl_inner["rho"] = inner_container_rho
        cyl_inner["l"] = h
        shapes.append(cyl_inner)

        # Generate random shapes
        def random_sphere():  # ruff: ignore[missing-return-type-private-function]
            posrad = np.sqrt(rng.random() * inner_radius**2)
            r = min(
                0.002 * inner_radius * np.exp(6 * rng.random() - 3),
                inner_radius - posrad,
            )
            # Choose whether this will be a potentially high-attenuation 'problem-maker' or a low-contrast object
            if rng.choice([0, 1], p=[0.90, 0.10]):
                rho = 0.1 * np.exp(6 * rng.random())
            else:
                rho = 1 + 0.10 * (np.exp(2 * rng.random()) - 1)
            phi = 2 * np.pi * rng.random()
            x = offset_x + posrad * np.cos(phi)
            y = offset_y + posrad * np.sin(phi)
            z = (h - 2 * r) * (rng.random() - 1 / 2)
            # Generate the shape baed on the randomised parameters
            t = "Sphere"
            s = np.array(0, dtype=fv.Shape)
            s["x"] = x
            s["y"] = y
            s["z"] = z
            s["r"] = r
            s["rho"] = rho
            s["type"] = fv._dict_shape_name_to_type_num[t]  # ruff: ignore[private-member-access]
            return s

        def random_cone():  # ruff: ignore[missing-return-type-private-function]
            axis_z = 1 - 2 * rng.random()
            axis_longitude = 2 * np.pi * rng.random()
            costheta = axis_z
            sintheta = np.sqrt(max(0, 1 - axis_z**2))
            axis = np.array(
                [
                    sintheta * np.cos(axis_longitude),
                    sintheta * np.sin(axis_longitude),
                    axis_z,
                ],
                dtype=np.float32,
            )  # In case of numerical imprecision, normalise the vector explcitly
            # Choose whether this will be a potentially high-attenuation 'problem-maker' or a low-contrast object
            if rng.choice([0, 1], p=[0.90, 0.10]):
                rho = 0.1 * np.exp(6 * rng.random())
            else:
                rho = 1 + 0.10 * (np.exp(3 * rng.random()) - 1)

            axis = axis / np.linalg.norm(axis)
            axis = np.array(tuple(axis), dtype=fv.fvec3)
            r1 = 0.02 * inner_radius * np.exp(4 * rng.random() - 2)
            r2 = 0.02 * inner_radius * np.exp(4 * rng.random() - 2)
            phi = 2 * np.pi * rng.random()
            posrad = np.sqrt(rng.random() * inner_radius**2)
            x = offset_x + posrad * np.cos(phi)
            y = offset_y + posrad * np.sin(phi)
            z = h * (rng.random() - 1 / 2)
            l = 10 * rng.random()
            # Cap the length so that the object doesn't pierce the inner cylinder
            try_count: int = 0
            while True:
                c1 = (
                    np.sqrt(
                        np.square(
                            posrad * np.cos(phi)
                            - l / 2 * np.cos(axis_longitude) * sintheta
                        )
                        + np.square(
                            posrad * np.sin(phi)
                            - l / 2 * np.sin(axis_longitude) * sintheta
                        )
                    )
                    + r1 * abs(costheta)
                ) + r1
                c2 = (
                    np.sqrt(
                        np.square(
                            posrad * np.cos(phi)
                            + l / 2 * np.cos(axis_longitude) * sintheta
                        )
                        + np.square(
                            posrad * np.sin(phi)
                            + l / 2 * np.sin(axis_longitude) * sintheta
                        )
                    )
                    + r2 * abs(costheta)
                ) + r2
                if max(c1, c2) <= inner_radius:
                    break
                if try_count > 10:
                    return None
                try_count += 1
                l *= 0.98
                r1 *= 0.95
                r2 *= 0.9

            # Generate the shape baed on the randomised parameters
            t = "Cone"
            s = np.array(0, dtype=fv.Shape)
            s["axis"] = axis
            s["x"] = x
            s["y"] = y
            s["z"] = z
            s["r1"] = r1
            s["r2"] = r2
            s["l"] = l
            s["rho"] = rho
            s["type"] = fv._dict_shape_name_to_type_num[t]  # ruff: ignore[private-member-access]
            # Apply clipping axes if necessary, to keep the cone within bounds
            if s["z"] + max(s["r1"], s["r2"]) + s["l"] / 2 > h / 2:
                # Add top clipping axis
                cas1 = np.array(0, dtype=fv.ClipAxisSpec)
                cas1["axis"] = np.array((0, 0, 1), dtype=fv.fvec3)
                cas1["dotprod"] = h / 2
                cas1["mode"] = 1
                s["cas1"] = cas1
            if s["z"] - max(s["r1"], s["r2"]) - s["l"] / 2 < -h / 2:
                # Add bottom clipping axis
                cas2 = np.array(0, dtype=fv.ClipAxisSpec)
                cas2["axis"] = np.array((0, 0, 1), dtype=fv.fvec3)
                cas2["dotprod"] = -h / 2
                cas2["mode"] = 2
                s["cas2"] = cas2
            return s

        def random_shape():  # ruff: ignore[missing-return-type-private-function]
            f = rng.choice(
                [
                    random_cone,
                    random_sphere,
                ],
                p=[0.3, 0.7],
            )  # ty is bugged
            return f()

        while len(shapes) < num_interior_objects + 2:
            shapes.extend(
                random_shape()
                for _ in range(num_interior_objects - len(shapes) + 2)
            )
            shapes = [s for s in shapes if s is not None]
        # Sort by largest radius to small-- this way, the big shapes won't
        # just overwrite the small ones
        shapes = sorted(
            shapes, key=lambda s: max(s["r"], s["r1"], s["r2"]), reverse=True
        )
        # Add the caps, if applicable
        if caps_height is not None:
            captop = np.array(0, dtype=fv.Shape)
            captop["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
            captop["rho"] = 1.1 * outer_container_rho
            captop["l"] = caps_height
            captop["r"] = outer_radius
            captop["x"] = offset_x
            captop["y"] = offset_y
            captop["z"] = h / 2 - caps_height / 2
            capbot = captop.copy()
            capbot["z"] = -h / 2 + caps_height / 2
            shapes.extend([captop, capbot])
        # Add the middle band. It will overwrite the interior shapes.
        mbh = middle_band_height or 0  # null-coalesce to 0
        if mbh > 0:
            mbz = 0

            mb_rho = 1.1 * outer_container_rho
            s = np.array(0, dtype=fv.Shape)
            s["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
            s["rho"] = mb_rho
            s["r"] = (inner_radius + outer_radius) / 2
            s["x"] = offset_x
            s["y"] = offset_y
            s["l"] = mbh
            s["z"] = mbz
            mbs = []
            mbs.append(s)
            # Make a high-frequency, low-contrast pattern in z.
            # This may be interfered with by the knurling.
            zs = np.linspace(-mbh / 2, mbh / 2, endpoint=True, num=11)
            dz = zs[1] - zs[0]
            zs = zs[2:-2]
            s = np.array(0, dtype=fv.Shape)
            s["type"] = fv._dict_shape_name_to_type_num["Ellipsoid"]  # ruff: ignore[private-member-access]
            s["x"] = offset_x + inner_radius / 2
            s["y"] = offset_y + inner_radius / 4
            s["rho"] = 1.05 * mb_rho
            s["dx"] = inner_radius / 3
            s["dy"] = inner_radius / 4
            s["dz"] = dz / 2
            for z in zs:
                s_ = s.copy()
                s_["z"] = z + mbz
                mbs.append(s_)
            # Make some large, high-frequency 'bubbles' in the band
            s = np.array(0, dtype=fv.Shape)
            s["type"] = fv._dict_shape_name_to_type_num["Sphere"]  # ruff: ignore[private-member-access]
            s["rho"] = 0
            r = min(0.0075 * inner_radius, mbh / 4)
            s["r"] = r
            numheight = min(7, max(1, mbh / (4 * r) - 2))
            xs = (np.arange(25) - (25 - 1) / 2) * 4 * r - inner_radius / 7
            ys = (np.arange(10) - (10 - 1) / 2) * 4 * r - inner_radius / 2
            zs = (np.arange(numheight) - (numheight - 1) / 2) * 4 * r
            for x, y, z in itertools.product(xs, ys, zs):
                s_ = s.copy()
                s_["x"] = offset_x + x
                s_["y"] = offset_y + y
                s_["z"] = z + mbz
                mbs.append(s_)
            shapes.extend(mbs)

        # Add the container knurling
        if num_knurls > 0:
            # Insert after the external container, but before the internal cylinder
            ch = caps_height or 0  # will null-coalesce to 0
            zs = np.linspace(
                -h / 2 + ch / 2,
                h / 2 - ch / 2,
                endpoint=True,
                num=2 + num_knurls,
            )
            dz = zs[1] - zs[0]
            # knurl_wid = dz
            zs = zs[1:-1]
            s = np.array(0, dtype=fv.Shape)
            s["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
            s["rho"] = 0
            s["x"] = offset_x
            s["y"] = offset_y
            s["l"] = dz / 2
            for z in zs:
                s_ = s.copy()
                s_["r"] = outer_radius
                s_["rho"] = outer_container_rho * (
                    1 + 0.02 * (1 - 2 * rng_knurl.random())
                )
                s_["z"] = z
                shapes.insert(0, s_)
            # Now add the cylindrical cut-outs
            r = 1 / 2 * (outer_radius - inner_radius)
            num_cutouts = max(
                3, int(np.ceil(outer_radius * 2 * np.pi / (2 * r) / 2))
            )  # arbitrarily chosen
            for theta in np.linspace(
                0, 2 * np.pi, num=num_cutouts, endpoint=False
            ):
                s = np.array(0, dtype=fv.Shape)
                s["type"] = fv._dict_shape_name_to_type_num["Cylinder_z"]  # ruff: ignore[private-member-access]
                s["rho"] = 0
                s["x"] = offset_x + outer_radius * np.cos(theta)
                s["y"] = offset_y + outer_radius * np.sin(theta)
                s["z"] = 0
                s["l"] = (
                    h - ch
                )  # have the cutouts go to half-way through the caps
                s["r"] = r
                shapes.append(s)

        return FBVolume(np.array(shapes, dtype=fv.Shape))


# ruff: enable[invalid-function-name,missing-trailing-period]
