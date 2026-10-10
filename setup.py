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

"""Standard setup file for the module."""

import os

from setuptools import setup

my_module_name = "aprict"
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
# Followed the advice in https://packaging.python.org/en/latest/guides/single-sourcing-package-version/
with open(os.path.join(my_module_dir, "version.py"), encoding="utf-8") as f:
    d = {}
    exec(f.read(), d)  # nosec # make bandit ignore this security issue
    version = d["version"]

setup(
    version=version,
    author="Murdock Grewar",
    # Using this approach makes pip fail to find subpackages and also the data directory.
    # packages=[my_module_name],
    # package_dir={my_module_name: "src/" + my_module_name},
    # package_data={my_module_name: ["data/"]},
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
