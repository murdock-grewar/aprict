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

"""
[![Static Badge](https://img.shields.io/badge/ORCiD-0000--0002--0015--8284-green?logo=orcid)](https://orcid.org/0000-0002-0015-8284)
[![Static Badge](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.23632625-blue?logo=DOI&color=blue)](https://doi.org/10.5281/zenodo.23632625)
![PyPI Version](https://img.shields.io/pypi/v/aprict)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

`aprict` is a Python module and commandline program for simulating flat-panel transmission X-ray experiments.

Recommended import pattern
--

The recommended import pattern for this module is
```python
import aprict
from aprict.cb import flatpanel as cbfp
```

Main components
--

The main components of the end-user Python API are:
 - [`aprict.FBVolume`](aprict/volume.html#FBVolume) - an attenuating volume defined in FORBILD format
 - [`aprict.Volume`](aprict/volume.html#Volume) - an attenuating volume defined in voxelised format
 - `aprict.cb.flatpanel` - a namespace for objects and functions applying to cone-beam imaging geometries

Examples
--

For examples of Python API usage, see `aprict.examples`.

Commandline interface
--

The entry point for the main commandline interface is the shell command `aprict`. This shell command should be available on `$PATH` if the module is installed in the current environment. All shell commands are self-documented:
```bash
$ aprict --help
```
"""  # ruff: ignore[ambiguous-unicode-character-docstring, blank-lines-between-header-and-content, mismatched-section-underline-length, missing-blank-line-after-summary, missing-trailing-period]

# If we are just running the build script, then I don't want to initialise anything,
# because the imports fail because it hasn't built yet.

import os as _os

if _os.environ.get("APRICT_BUILDMODE", ""):  # ruff: ignore[non-empty-init-module]
    pass
else:
    # Ensure that numexpr max threads is set to the number of logical cores before importing numexpr either directly or indirectly through other imports.
    _nethreads_max_env = _os.environ.get("NUMEXPR_MAX_THREADS", None)
    if _nethreads_max_env is None:
        _os.environ["NUMEXPR_MAX_THREADS"] = str(_os.process_cpu_count())

    del _os
    from . import cb as cb
    from . import image_view as imview
    from . import iterators, lib, phantoms, volume
    from .cuda import structs as dtypes
    from .volume import FBVolume as FBVolume

    FBVolume.__doc__ = "@private\nAn alias to aprict.volume.FBVolume"  # Hide from pdoc documentation of this module
    from .volume import Volume as Volume

    Volume.__doc__ = "@private\nAn alias to aprict.volume.Volume"  # Hide from pdoc documentation of this module

    # Note that __all__ determines what's listed in the documentation page.
    # So, I'm selecting only those parts which I want to be documented on the main doc page here.
    __all__ = [
        "FBVolume",  # documented in aprict.volume
        "Volume",  # documented in aprict.volume
        "cb",
        "dtypes",
        "imview",
        "iterators",
        "lib",
        "phantoms",
        "volume",
    ]
