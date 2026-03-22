# setup.py
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

"""Setup file for the module."""

import os
import runpy

from setuptools import setup

my_module_name = "snookerr"
my_src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
my_module_dir = os.path.join(my_src_dir, my_module_name)
my_script_dir = os.path.join(my_module_dir, "scripts")
my_scripts = [
    script_name
    for script_name in os.listdir(my_script_dir)
    if script_name.endswith(".py") and not script_name.startswith("__")
]


def _remove_extension_from_filename(script_name: str) -> str:
    return ".".join(script_name.split(".")[:-1])


def _filename_to_command_name(script_name: str) -> str:
    return _remove_extension_from_filename(script_name).replace("_", "-")


# Get the version number
version: str = runpy.run_path(os.path.join(my_module_dir, "version.py"))[
    "version"
]

setup(
    version=version,
    entry_points={
        "console_scripts": [
            my_module_name
            + "-"
            + f"{_filename_to_command_name(script_name):s}"
            + " = "
            + my_module_name
            + f".scripts.{_remove_extension_from_filename(script_name):s}:main"
            for script_name in my_scripts
        ],
    },
)
