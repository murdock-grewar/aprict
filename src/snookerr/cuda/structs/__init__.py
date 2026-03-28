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

"""Repository of KernelStruct objects to generate .h files from.

The _generate_headers.py will mirror the directory structure of this
directory. Each *.py file in this directory (except for any called
__init__.py) will be opened, and a C struct produced from each of them.

Each matched 'filename.py' file is expected to contain a KernelStruct object
with the name 'filename'. The 'struct_name' field of the object should also
by 'filename'.

The _generate_headers.py will call 'filename.save_c_struct_to_file(...)' and
will generate the .h files in a directory structure that mirrors this
directory.
"""

from . import test_subdir as test_subdir
from .base import *
from .config_cb_project import *
from .geometry import *
