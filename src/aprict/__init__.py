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

r"""
[![Static Badge](https://img.shields.io/badge/Author_ORCiD-0000--0002--0015--8284-green?logo=orcid)](https://orcid.org/0000-0002-0015-8284)
[![Static Badge](https://img.shields.io/badge/Project_DOI-10.5281%2Fzenodo.23632625-yellow?logo=DOI&color=yellow)](https://doi.org/10.5281/zenodo.23632625)
![PyPI Version](https://img.shields.io/pypi/v/aprict)
[![Project Status: WIP – Initial development is in progress, but there has not yet been a stable, usable release suitable for the public.](https://www.repostatus.org/badges/latest/wip.svg)](https://www.repostatus.org/#wip)

# What is `aprict`?

`aprict` is a Python module for simulating and reconstructing from flat-panel transmission X-ray experiments. In other words, it is a library for Cone-Beam Computed Tomography (CBCT).

## Features

`aprict` uses the GPU for all the heavy operations. The terminology 'in-core' and 'out-of-core' here refers to whether the relevant data must fit entirely within the GPU VRAM (in-core) or not (out-of-core). Most operations have no minimum RAM requirement in practice. Included features as of `aprict==0.0.33`:
 - Out-of-core calculation of projection data for arbitrary flat-panel geometries. Supports voxelised volumes and (almost) any FORBILD phantom. (Exact line integrals in all cases.)
 - Out-of-core analytic reconstruction for special geometries.
 - In-core iterative reconstruction for arbitrary geometries.

## Installation

You can install `aprict` with `pip install aprict[gui]`.

## Quickstart

Follow this three-step process to get a quick flavour of the capabilities of this package. Run these commands in a shell such as `bash`, from a fresh directory, with an active Python environment with `aprict[gui]` installed. Note that all units of distance are measured in millimeters.

 1. Create a `projgeom.json` file encoding the experiment geometry:
    ```bash
    aprict make-geom cone-cyl \
        --num 1000 \
        --type GOLDEN_THETA \
        --R 250 --L-minus-R 250 --h 700 \
        --detwid 600 --dethei 600 \
        --pixwid-isoc 0.25 --pixhei-isoc 0.25 \
        --output projgeom.json
    ```
 2. Create the linearised projection data in directory `projs/`, based on the geometry encoded in `projgeom.json`:
    ```bash
    aprict project cbfp \
        --geom projgeom.json \
        --phantom-preset HEAD \
        --ssu 3 --ssv 3 \
        --output projs \
        --plot-mode PYQTGRAPH  # available if `gui` extra installed
    ```
 3. Create a tomographic reconstruction in directory `recon_gbc/`, from the geometry and simulated data. Uses the Global Backprojection-Convolution (GBC) algorithm:
    ```bash
    aprict recon gbc-cone-cyl \
        --geom projs/projgeom.json \
        --projs projs/projs.npy \
        --voxwid 0.5 \
        --volshape '(525,525,525)' \
        --alg GREWAR \
        --f1 1.2 --f2 6 --n 9 --eps 1e-9 \
        --plotmin 0 --plotmax 0.0368 \
        --output recon_gbc \
        --plot-mode PYQTGRAPH  # available if `gui` extra installed
    ```
    PNG previews of the reconstruction are available at `recon_gbc/output/*.png`. The full reconstruction is available at `recon_gbc/output/recon.npy`.

Optionally, the reconstruction can be converted to `OME-Zarr 5.0` format with:
```bash
aprict convert npy_to_omez \
    --input recon_gbc/output/recon.npy \
    --output ./recon.zarr \
    --voxwid-mm 0.5 \
    --ver V0_5
```

# Recommended use

## Commandline interface (CLI)

The entry point for the main commandline interface is the shell command `aprict`. This shell command should be available on `$PATH` if the module is installed in the current environment. All shell commands are self-documented:
```bash
$ aprict --help
```

## Python API
### Recommended import pattern

The recommended import pattern for this module is
```python
import aprict
from aprict.cb import flatpanel as cbfp
```

### Main components of the Python API

The main components of the end-user Python API are:
 - [`aprict.FBVolume`](aprict/volume.html#FBVolume) - an attenuating volume defined in FORBILD format
 - [`aprict.Volume`](aprict/volume.html#Volume) - an attenuating volume defined in voxelised format
 - `aprict.cb.flatpanel` - a namespace for objects and functions applying to cone-beam imaging geometries

### Examples using the Python API

For examples of Python API usage, see `aprict.examples`.

"""  # ruff: ignore[ambiguous-unicode-character-docstring, missing-blank-line-after-summary, missing-trailing-period]

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
