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

"""Module to provide an do-nothing context manager."""


class EmptyContextManager:
    """A context manager that does nothing."""

    def __enter__(self) -> None:  # ruff: ignore[undocumented-magic-method]
        pass

    def __exit__(self, exc_type, exc_value, exc_tb) -> None:  # ruff: ignore[missing-type-function-argument, undocumented-magic-method]
        pass
