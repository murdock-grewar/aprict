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

import pathlib

from .. import data
from ..lib import process_pha_file as ppf
from ..volume import FBVolume

path_to_FORBILD_src = pathlib.Path(str(data.__path__[0])).joinpath("FORBILD")  # noqa: N816


# ruff: disable[N802,D400]


class _MetaPhantomContainer(type):
    @property
    def HeadPhantom(cls) -> FBVolume:
        """HeadPhantom.pha"""
        return FBVolume(
            ppf.convert_ph_file_to_shapes(
                path_to_FORBILD_src.joinpath("HeadPhantom.pha")
            )
        )


class PhantomContainer(metaclass=_MetaPhantomContainer):
    """A container for the phantom objects."""


# ruff: enable[N802,D400]
