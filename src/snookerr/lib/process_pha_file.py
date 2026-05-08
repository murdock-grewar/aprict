#!/usr/bin/env python
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

"""A hacked together script to convert some .pha files to a list of Shapes for use in snookerr."""

import functools
import io
import itertools
import multiprocessing
import operator
import re
import sys
from typing import TYPE_CHECKING

import numexpr as ne
import numpy as np
import pcpp
import regex  # replaces standard 're' module. Needed for the ability to do balanced matching of parentheses

from ..cuda.structs import FORBILD_volumes as fv

if TYPE_CHECKING:
    import pathlib

_typedict = {
    "Sphere": fv.Sphere,
    "Box": fv.Box,
    "Cylinder_x": fv.Cylinder_x,
    "Cylinder_y": fv.Cylinder_y,
    "Cylinder_z": fv.Cylinder_z,
    "Cylinder": fv.Cylinder,
    "Ellipsoid": fv.Ellipsoid,
    "Ellipsoid_free": fv.Ellipsoid_free,
    "Ellipt_Cyl": fv.Ellipt_Cyl,
    "Cone": fv.Cone,
    "Cone_x": fv.Cone_x,
    "Cone_y": fv.Cone_y,
    "Cone_z": fv.Cone_z,
}

re_obj = re.compile(r"{[^}]*}")
re_key = re.compile(r"(?<=[ \[])[^= ]+=\s*[^ ]+(?=[ \]])")
re_shape = re.compile(r"\[[^\]]*\]")
re_shape_name = re.compile(r"(?<=[\[ ])[^ \]:]*(?=:)")
re_clip_plane = re.compile(r"(?<=[\[ \{])[^ <>:=]+[<>][^ <>:=]+(?=[ \]\}])")
# re_define_pragma = re.compile(r"\s*#define\s+([^\s]+)\s+([^\s].*)")
re_float3 = re.compile(
    r"(?<=[\[\{ ])([^ <>:=,()]+)\(([^ <>:=,]+?),([^ <>:=,]+?),([^ <>:=,]+)\)"  # assume no spaces inside (...)
    # r"(?<=[\[\{ ])([^ <>:=,()]+)\(([^<>:=,]+?),([^<>:=,]+?),([^<>:=,]+)\)"  # works for lungs phantom
    # r"(?<=[\[\{ ])([^ <>:=,()]+)\(([^<>:=,]+?),([^<>:=,]+?),([^ <>:=,]+)\)"  # works for the head phantom
)


def _preprocess_file(fn: str | pathlib.Path) -> str:
    p = pcpp.Preprocessor()
    with open(fn, "r", encoding="ISO-8859-1") as f:
        p.parse(f)
    oh = io.StringIO()
    p.write(oh)
    return oh.getvalue()


# Slower than I would like
def _preprocess_str(s: str) -> str:
    p = pcpp.Preprocessor()
    p.parse(s)
    oh = io.StringIO()
    p.write(oh)
    return oh.getvalue()


def convert_ph_file_to_shapes(  # noqa: D103
    fn: str | pathlib.Path,
) -> np.ndarray[np.dtype[fv.Shape]]:  # ty:ignore[invalid-type-arguments]
    return _convert_preprocessed_ph_str_to_shapes(_preprocess_file(fn))


def convert_ph_str_to_shapes(  # noqa: D103
    s: str, skip_preprocessing: bool = False
) -> np.ndarray[np.dtype[fv.Shape]]:  # ty:ignore[invalid-type-arguments]
    if skip_preprocessing:
        return _convert_preprocessed_ph_str_to_shapes(s)
    return _convert_preprocessed_ph_str_to_shapes(_preprocess_str(s))


def _convert_preprocessed_ph_str_to_shapes(
    s: str,
) -> np.ndarray[np.dtype[fv.Shape]]:  # ty:ignore[invalid-type-arguments]

    # Now get rid of newlines. Treat it as one long string
    # Shapes are contained in { ... } which makes it easy to process
    s = s.replace("\n", " ")

    # translations
    s = s.replace("achse", "axis")
    s = s.replace("dichte", "rho")
    s = re.sub(
        r"(cos|sin|tan)\(([\-+*/0-9\. ]*)\)",
        lambda m: (
            m.groups()[0]
            + "("
            + "3.1415926535898/180*("
            + m.groups()[1]
            + "))"
        ),
        s,
    )
    # remove redundant "mm" units for distances
    s = re.sub(
        r"([0-9\.])mm",
        lambda m: m.groups()[0],
        s,
    )
    # remove "Detector{...}" specifications. We only want the phantom.
    s = re.sub(
        r"Detector\{[^\{\}]*\}",
        lambda _: "",
        s,
    )
    # remove space between "axis" and opening parenthesis. Do this for each vector type
    s = re.sub(
        r"(axis|a_x|a_y|a_z) \(",
        lambda m: m.groups()[0] + "(",
        s,
    )
    objs = re_obj.findall(s)
    min_shapes_per_group: int = 300
    nprocs = min(
        (len(objs) + (min_shapes_per_group - 1)) // min_shapes_per_group, 64
    )
    groupsize = (len(objs) + (nprocs - 1)) // nprocs
    if len(objs) <= groupsize:
        shapes = _objs_to_shape(objs)
    else:
        with multiprocessing.Pool(nprocs) as p:
            shapes = functools.reduce(
                operator.iadd,
                p.map(_objs_to_shape, itertools.batched(objs, groupsize)),
                [],
            )
    return np.array(shapes, dtype=fv.Shape, ndmax=1)


# To make the re_float3 regex pattern work, remove spaces from any expression in parentheses
def _remove_spaces_from_parentheticals(s: str) -> str:
    return regex.sub(
        r"(?V1)\(((?:[^)(]+|(?R))*+)\)",  # the "(?V1)" is a flag for the 'regex' package. It's needed for the recursion feature.
        lambda m: "(" + m.groups()[0].replace(" ", "") + ")",
        s,
    )


# Need this to be at the top-level so it can be used in multiprocessing.Pool.map
def _objs_to_shape(objs):  # noqa: ANN001, ANN202, C901
    shape_objs = []
    for obj in objs:
        obj = _remove_spaces_from_parentheticals(obj)  # noqa: PLW2901
        shape = re_shape.findall(obj)[0]
        remaining = re_shape.sub("", obj)
        remaining_properties = re_key.findall(remaining)
        shape_name = re_shape_name.findall(shape)[0]
        shape_properties = re_key.findall(shape)
        clip_planes = re_clip_plane.findall(shape)

        # Construct an empty object of the correct type
        shape_obj = np.array(0, dtype=fv.Shape)
        shape_obj["type"] = fv._dict_shape_name_to_type_num[shape_name]  # noqa: SLF001
        # Fill the shape with the found properties
        for k, v in [
            prop.split("=") for prop in shape_properties + remaining_properties
        ]:
            if k in ("formel", "formula"):
                continue
            val = ne.evaluate(v.strip())
            shape_obj[k] = val
        for m in re_float3.finditer(shape):
            name, x, y, z = m.groups()
            x, y, z = [ne.evaluate(thing.strip()) for thing in (x, y, z)]
            vec = np.array((x, y, z), dtype=fv.fvec3)
            shape_obj[name] = vec

        # The FORBILD format only requires the user to specify two axes, if they wish
        # The third must be computed from the other two.
        zerofvec3 = np.array(0, dtype=fv.fvec3)
        match shape_name:
            case "Ellipsoid_free":
                vec_name_set = {"a_x", "a_y", "a_z"}
                for vecname in vec_name_set:
                    if shape_obj[vecname] == zerofvec3:
                        u, v = [
                            shape_obj[vn] for vn in vec_name_set - {vecname}
                        ]
                        u = np.array(
                            [u["x"], u["y"], u["z"]], dtype=np.float32
                        )
                        v = np.array(
                            [v["x"], v["y"], v["z"]], dtype=np.float32
                        )
                        shape_obj[vecname] = np.array(
                            tuple(np.cross(u, v)), dtype=fv.fvec3
                        )
                        break
            case "Ellipt_Cyl":
                vec_name_set = {"a_x", "a_y", "axis"}
                for vecname in vec_name_set:
                    if shape_obj[vecname] == zerofvec3:
                        u, v = [
                            shape_obj[vn] for vn in vec_name_set - {vecname}
                        ]
                        u = np.array(
                            [u["x"], u["y"], u["z"]], dtype=np.float32
                        )
                        v = np.array(
                            [v["x"], v["y"], v["z"]], dtype=np.float32
                        )
                        shape_obj[vecname] = np.array(
                            tuple(np.cross(u, v)), dtype=fv.fvec3
                        )
                        break

        # Process the clipping planes, up to an allowed maximum
        clip_plane_counter: int = 0
        for sign, (k, v) in [
            ("<", prop.split("<")) for prop in clip_planes if "<" in prop
        ] + [(">", prop.split(">")) for prop in clip_planes if ">" in prop]:
            match k:
                case "x":
                    axis = np.array((1, 0, 0), dtype=fv.fvec3)
                case "y":
                    axis = np.array((0, 1, 0), dtype=fv.fvec3)
                case "z":
                    axis = np.array((0, 0, 1), dtype=fv.fvec3)
                case _:
                    raise NotImplementedError()
            match sign:
                case "<":
                    mode = 1
                case ">":
                    mode = 2
                case _:
                    raise NotImplementedError()
            dotprod = ne.evaluate(v)
            cas = np.array((axis, dotprod, mode), dtype=fv.ClipAxisSpec)
            clip_plane_counter += 1
            match clip_plane_counter:
                case 1:
                    shape_obj["cas1"] = cas
                case 2:
                    shape_obj["cas2"] = cas
                case 3:
                    shape_obj["cas3"] = cas
                case _:
                    msg = "Too many clip planes encountered."
                    raise ValueError(msg)
        shape_objs.append(shape_obj)
    return shape_objs


if __name__ == "__main__":
    ans = convert_ph_file_to_shapes(sys.argv[1])
    print(ans)
