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

"""Tools to deal with importing packages with anti-patterns."""


def import_neuroglancer():  # ruff: ignore[missing-return-type-undocumented-public-function]
    """Import neuroglancer and remove the handlers it adds to the root logger."""
    # Neuroglancer adds a root logging handler, which I did not ask for. Need to import it and then remove the logger it added.
    import logging  # ruff: ignore[import-outside-top-level]

    handlers = logging.root.handlers[
        :
    ]  # I don't want to copy a reference to the list, I want to construct a new list with the current contents. Hence the [:].
    import neuroglancer  # ruff: ignore[import-outside-top-level]

    for handler in logging.root.handlers[:]:
        if handler not in handlers:
            logging.root.removeHandler(handler)
    return neuroglancer


def purge_root_loggers() -> None:
    """Purge all handlers of the root logger."""
    import logging  # ruff: ignore[import-outside-top-level]

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
