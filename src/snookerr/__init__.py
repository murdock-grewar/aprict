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

"""Main module."""

# If we are just running the build script, then I don't want to initialise anything,
# because the imports fail because it hasn't built yet.

import os as _os

if _os.environ.get("SNOOKERR_BUILDMODE", ""):  # noqa: RUF067
    pass
else:
    del _os
    from . import cb as cb
    from . import image_view as imview
    from . import iterators, lib
    from .cuda import structs as dtypes
    from .volume import Volume

    __all__ = [
        "Volume",
        "cb",
        "dtypes",
        "imview",
        "iterators",
        "lib",
    ]
