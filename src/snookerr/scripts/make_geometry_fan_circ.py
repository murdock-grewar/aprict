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

"""Produce a ProjGeom object for a circular fan-beam scan, and save it as a .json file."""

import enum
import pathlib
import sys
import types

import click
import click_extra as ce
import cloup
import numpy as np

from snookerr.lib import click_tools

from ..cb import flatpanel as cbfp

_progdesc = (
    "Produce a ProjGeom object for a circular fan-beam scan, and save it as a .json file."
)


class TYPE(enum.Enum):  # ruff: ignore[undocumented-public-class]
    EQUISPACED = 1
    GOLDEN = 2


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
    "--type",
    type=click.Choice(TYPE),
    default=TYPE.EQUISPACED,
    help="How the source points should be distributed.",
)
@ce.option(
    "--num", required=True, type=click.INT, help="Number of projections."
)
@ce.option(
    "--output",
    "-o",
    required=True,
    # Note that setting file_okay=False and dir_okay=False causes it to only accept uncreated paths
    type=click.Path(
        path_type=pathlib.Path,
        writable=True,
        file_okay=False,
        dir_okay=False,
        allow_dash=True,
    ),
    help="Path at which to save the .json. (Use a dash to print to stdout)",
)
@ce.option_group(
    "detector pixel width",
    ce.option(
        "--pixwid",
        type=click.FLOAT,
        help="Width of detector pixels.",
    ),
    ce.option(
        "--pixwid-isoc",
        type=click.FLOAT,
        help="Width of detector pixels, when projected to the isocenter.",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option_group(
    "detector pixel height",
    ce.option(
        "--pixhei",
        type=click.FLOAT,
        help="Height of detector pixels.",
    ),
    ce.option(
        "--pixhei-isoc",
        type=click.FLOAT,
        help="Height of detector pixels, when projected to the isocenter.",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option_group(
    "detector width",
    ce.option(
        "--nu",
        type=click.INT,
        help="Width of detector, in pixels.",
    ),
    ce.option(
        "--detwid",
        type=click.FLOAT,
        help="Physical width of detector.",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option_group(
    "detector height",
    ce.option(
        "--nv",
        type=click.INT,
        help="Height of detector, in pixels.",
    ),
    ce.option(
        "--dethei",
        type=click.FLOAT,
        help="Physical height of detector.",
    ),
    constraint=cloup.constraints.RequireExactly(1),
)
@ce.option(
    "--isocenter",
    "--isocentre",
    default=(0.0, 0.0, 0.0),
    type=click_tools.FloatTupleArgTypeFactory(numfloats=3),
    help="Location of the centre of the circle.",
)
@ce.option(
    "--R",
    "R",  # makes it go into args.R instead of args.r
    required=True,
    type=click.FLOAT,
    help="Radius of circle.",
)
@ce.option(
    "--L-minus-R",
    "L_minus_R",  # makes it go into args.L_minus_R instead of args.l_minus_r
    required=True,
    type=click.FLOAT,
    help="Distance from isocentre to the middle of the detector.",
)
@ce.option(
    "--clockwise",
    type=click_tools.BoolArgType(),
    default=False,
)
@ce.option(
    "--theta-start",
    type=click_tools.MaybeType(click.FLOAT),
    default=None,
    help="Angle, in radians, of the first source point. Leave this at None to select it randomly.",
)
@ce.option_group(
    "random seeds",
    ce.option(
        "--seed-starting-theta",
        type=click_tools.MaybeType(click.INT),
        help="Seed to use for choosing the starting theta.",
    ),
    help="Seeds to use for random number generation. Leave an option as None to use a pseudorandom seed.",
)
def main(**kwargs) -> None:  # ruff: ignore[missing-type-kwargs]
    """Parse input arguments using click."""
    args = types.SimpleNamespace(**kwargs)
    main_from_args(args)


def main_from_args(args: types.SimpleNamespace) -> None:  # ruff: ignore[undocumented-public-function]

    if args.nu is not None and args.nu < 1:
        msg = "--nu must be a positive integer."
        raise ValueError(msg)
    if args.nv is not None and args.nv < 1:
        msg = "--nv must be a positive integer."
        raise ValueError(msg)
    if args.pixwid_isoc is not None and args.pixwid_isoc <= 0:
        msg = "--pixwid must be a positive number."
        raise ValueError(msg)
    if args.pixhei_isoc is not None and args.pixhei_isoc <= 0:
        msg = "--pixhei must be a positive number."
        raise ValueError(msg)
    if args.pixwid is not None and args.pixwid <= 0:
        msg = "--pixwid must be a positive number."
        raise ValueError(msg)
    if args.pixhei is not None and args.pixhei <= 0:
        msg = "--pixhei must be a positive number."
        raise ValueError(msg)

    pixwid: float = args.pixwid or args.pixwid_isoc * (
        1 + args.L_minus_R / args.R
    )
    pixhei: float = args.pixhei or args.pixhei_isoc * (
        1 + args.L_minus_R / args.R
    )
    nu: int = args.nu or max(1, round(args.detwid / pixwid))
    nv: int = args.nv or max(1, round(args.dethei / pixhei))
    rng_starting_theta: np.random.Generator = np.random.default_rng(
        args.seed_starting_theta
    )
    theta_start: float = (
        args.theta_start
        if args.theta_start is not None
        else rng_starting_theta.random() * 2 * np.pi
    )

    # Make geometry
    zs = np.zeros((args.num,), dtype=np.float64)
    match args.type:
        case TYPE.EQUISPACED:
            ts = np.linspace(
                theta_start,
                theta_start + 2 * np.pi * (-1 if args.clockwise else 1),
                endpoint=False,
                num=args.num,
                dtype=np.float64,
            )
        case TYPE.GOLDEN:
            ts = (
                theta_start
                + (-1 if args.clockwise else 1)
                * np.arange(args.num, dtype=np.float64)
                * (np.sqrt(5) - 1)
                * np.pi
            )
        case _:
            raise NotImplementedError()
    source_pos_arr = np.array(
        [args.R * np.cos(ts), args.R * np.sin(ts), zs], dtype=np.float64
    ).transpose()
    det_pos_arr = np.array(
        [-args.L_minus_R * np.cos(ts), -args.L_minus_R * np.sin(ts), zs],
        dtype=np.float64,
    ).transpose()
    # Translate to isocenter position
    source_pos_arr += np.array(args.isocenter)[np.newaxis, :]
    det_pos_arr += np.array(args.isocenter)[np.newaxis, :]
    # Define pixel coordinate vectors
    det_uvec_arr = (
        pixwid
        * np.array(
            [-np.sin(ts), np.cos(ts), 0 * ts], dtype=np.float64
        ).transpose()
    )
    det_vvec_arr = pixhei * np.array([[0, 0, -1]], dtype=np.float64)
    pg: cbfp.ProjGeom = cbfp.ProjGeom(
        source_pos=source_pos_arr,
        det_pos=det_pos_arr,
        det_uvec=det_uvec_arr,
        det_vvec=det_vvec_arr,
        nu=nu,
        nv=nv,
    )
    json_str: str = pg.json_str()

    # Write the JSON output
    fp = (
        sys.stdout
        if str(args.output) == "-"
        else open(args.output, mode="w+", encoding="utf-8")
    )
    fp.write(json_str + "\n")
    fp.close()  # Don't print anything to stdout after this point, because the file pointer is closed
