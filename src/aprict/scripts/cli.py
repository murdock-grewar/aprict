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

"""The main command-line interface to aprict."""

import click_extra as ce

from ..lib import click_tools
from ..version import version
from .convert_npy_to_omezarr import main as main_convert_npy_to_omez
from .convert_omezarr_to_npy import main as main_convert_omez_to_npy
from .make_geometry_cone_cyl import main as main_make_geom_cone_cyl
from .make_geometry_fan_circ import main as main_make_geom_fan_circ
from .project_cbfp import main as main_project_cbfp
from .recon_cbfp_cyl_fbp import main as main_fbp
from .recon_cbfp_cyl_gbc import main as main_gbc

_progdesc: str = f"Entry point for CLI interface to aprict version {version}"


_cs = {
    "show_default": True,
    "show_choices": True,
    "terminal_width": None,
    "max_content_width": 300,
}


@ce.group(name="aprict", params=None, help=_progdesc, context_settings=_cs)
@ce.version_option(version=version)  # add --version from click_extra
@ce.tree_option()  # add --tree option from click_extra
# not including the man_option because it includes incorrect information, like possible exit codes etc.
# I could refine the subcommands etc in the future and re-enable this.
# @ce.man_option()  # add --man option from click_extra
def main() -> None:  # ruff: ignore[undocumented-public-function]
    pass


@ce.group(
    name="convert",
    params=None,
    help="Conversion between data formats.",
    context_settings=_cs,
)
def convert() -> None:  # ruff: ignore[undocumented-public-function]
    pass


main.add_command(convert)
convert.add_command(cmd=main_convert_npy_to_omez, name="npy_to_omez")
convert.add_command(cmd=main_convert_omez_to_npy, name="omez_to_npy")


@ce.group(
    name="recon",
    params=None,
    help="Tomographic reconstruction routines.",
    context_settings=_cs,
)
def recon() -> None:  # ruff: ignore[undocumented-public-function]
    pass


main.add_command(recon)

main_fbp_fan = click_tools.command_with_option_substitutions(
    main_fbp,
    name="fbp-fan-circ",
    help="Filtered backprojection for a fan-beam geometry with a circular source locus.",
    fixed_option_values={
        "fan_beam": True,
        "limit_zeta": None,
        "num_rot_angs": None,
        "ssz": 1,
    },
)
recon.add_command(cmd=main_fbp_fan)

main_fbp_cone = click_tools.command_with_option_substitutions(
    main_fbp,
    name="fbp-cone-cyl",
    help="Filtered backprojection for a cone-beam geometry with a cylindrical source locus.",
    fixed_option_values={
        "fan_beam": False,
    },
)
recon.add_command(cmd=main_fbp_cone)

main_gbc_cone = click_tools.command_with_option_substitutions(
    main_gbc,
    name="gbc-cone-cyl",
    help="Global Backprojection-Convolution for a cone-beam geometry with a cylindrical source locus.",
    fixed_option_values={},
)
recon.add_command(cmd=main_gbc_cone)


@ce.group(
    name="project",
    params=None,
    help="Generate projection data.",
    context_settings=_cs,
)
def project() -> None:  # ruff: ignore[undocumented-public-function]
    pass


main.add_command(project)

project.add_command(cmd=main_project_cbfp, name="cbfp")


@ce.group(
    name="make-geom",
    params=None,
    help="Make a geometry object.",
    context_settings=_cs,
)
def make_geom() -> None:  # ruff: ignore[undocumented-public-function]
    pass


main.add_command(make_geom)

make_geom.add_command(cmd=main_make_geom_fan_circ, name="fan-circ")
make_geom.add_command(cmd=main_make_geom_cone_cyl, name="cone-cyl")
