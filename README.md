<details open><summary><h1>About</h1></summary>

Description
---

`aprict` is a Python module for simulating transmission experiments in X-ray Computed Tomography, and for performing reconstructions from real and simulated data. In addition to the Python API, there is a commandline interface for some common operations.

Disclaimer
---

This software is provided with no warranty or guarantee of correctness. This is my research code sandbox, subject to breaking changes on a whim. This is not mature software. So, if you like it and wish to use it, then it is best to download a fixed version and use that. Updates are very likely to break things.

Hardware Requirements
---

This software requires at least one relatively recent *CUDA-capable GPU*. An RTX 3090 is recent enough. The relevant drivers will also need to be installed. You may need up-to-date drivers, depending on when the code was last updated.

It is strongly recommended to use high-speed mass storage, such as an NVMe drive. The software has not been tested on slower mass storage, and is not expected to perform well without it.

Known limitations
---

* Only flat-panel X-ray data with square pixels is supported, at this time.
* Lack of documentation:
  * The Python API is not yet documented.
  * The way that geometry is encoded in `*.json` files is not yet documented.
  * The encoding of projection data and volume data is not yet documented. (Numpy arrays; memory-mapped in the case of projections.)
* Implementation-related issues which could be alleviated in the future:
  * Volume size is limited by CUDA texture limit. For example, a `(2400,2400,4800)` volume can be reconstructed, but not a `(30000,30000,1)` volume, due to a volume of this size not fitting within a single CUDA texture
  * The FORBILD shape `Tetrahedron` is not yet implemented, and no primitive shape is permitted to have more than 3 clipping planes applied, due to internal limitations. All other FORBILD phantoms should be supported.
  * Only one GPU is utilised by this module, at the present time. There is no defined behaviour for which GPU will be chosen.
  * Not fully optimised: among other issues, 
    * the GBC reconstruction script assumes that the whole volume (or more) can fit in RAM. This limitation may be lifted in the future. (Note: a present workaround is to create a large swapfile on disk, since swap space is treated as RAM when the physical RAM is exhausted.)
  * CPU and GPU work not completely asynchronous: due to unresolved bugs, various asynchronous GPU operations have been disabled. This may not have much impact on large reconstructions, but could result in poor GPU utilisation for smaller reconstructions.
  * No smart batching of (back)projections: smarter code would analyse the geometry of projections, and batch together (back)projections that involve similar regions of the volume. Currently, the code does not do any kind of clever batching like this, and so the (back)projection code is not as performant as it could be.

Features
---

* Simulation of projection data for almost any flat-panel geometry, with almost any FORBILD phantom. Also supports voxelised volumes. The projection code computes exact line integrals (including through voxelised volumes composed of voxels of uniform attenuation coefficient), and can accept arbtirary phantoms defined in `*.pha` files using the FORBILD format. The accuracy can be further increased by averaging multiple line integrals for each pixel. The source and detector can even be placed _inside_ the object, and projection will still be accurately computed from the source point to the detector pixels.
* Several built-in phantoms included for the user's convenience.
* GBC and FBP reconstruction scripts supplied. These work efficiently with projection datasets of any size, e.g. a 200 GiB+ set is OK. (Requires high-speed mass storage for rapid reading/writing, e.g. an NVMe drive.) 

* Analytic reconstruction from:
  * Fan-beam data with circular scan
  * Cone-beam data with cylindrical scan

</details>

<details><summary><h1>Setup</h1></summary>

Installation
---

You can install `aprict` like you would any python package, using `pip install`. It is recommended to install it with the `[gui]` extra if possible, as this will enable plotting capabilities. This may require `PyQt6` or `pyside` or some other GUI backend to be installed on your system. Try installing with `[gui]` first, if you can. This README will assume that the `[gui]` extra is installed. 

Example installation using `uv`: 
```bash
$ uv venv -p 3.14
$ uv pip install aprict[gui]==0.0.19
$ source .venv/bin/activate
```
You should now have several executables available on `$PATH`. Try `aprict-cli --help` to see the polished commandline interface. 

If you're using Windows, then it's recommended to install this inside the Windows Subsystem for Linux (WSL).
</details>


<details><summary><h1>Example usage with the Commandline Interface (CLI)</h1></summary>


Commandline interface (CLI)
---

Activate the environment:
```bash
source .venv/bin/activate
```
Now, try the shell command `aprict-cli --help`. Here is the output as at `v0.0.19`:
```
Usage: aprict-cli [OPTIONS] COMMAND [ARGS]...

  Entry point for aprict CLI.

Options:
  --version   Show the version and exit.
  --tree      Show the tree of nested subcommands and exit.
  -h, --help  Show this message and exit.

Commands:
  help       Show help for a command.
  make-geom  Make a geometry object.
  project    Generate projection data.
  recon      Tomographic reconstruction routines.
```

You can use the `--help` option with any command to see its detailed usage information.


<details><summary><h2>Example A: cone-beam; cylindrical source distribution</h2></summary>

Here's an example where we will use the CLI to simulate data from the FORBILD Head phantom, and reconstruct a tomogram from that data.

Start by creating a clean directory to work in:
```bash
mkdir headtest
cd headtest
```

Now, the first step is to create a `.json` file representing the experiment geometry. This can be done like so:
```bash
aprict-cli make-geom cone-cyl \
    --num 2000 \
    --R 250 --L-minus-R 250 --h 700 \
    --detwid 600 --dethei 600 \
    --pixwid-isoc 0.25 --pixhei-isoc 0.25 \
    --output projgeom.json
```
You might like to open this file in a text editor, to get a feel for what's inside. It describes an experiment geometry with a flat-panel trajectory. 

Now, let's generate accurate, noiseless projection data from this geometry. We can specify any FORBILD file, assumed to have a file suffix of `.pha`, using the `--phantom file.pha` option. However, there are also a few built-in presets for convenience. Let's use `--phantom-preset HEAD`, which refers to the FORBILD Head phantom, with distances specified in units of millimeters.
```bash
aprict-cli project cbfp \
    --geom projgeom.json \
    --phantom-preset HEAD \
    --ssu 3 \
    --ssv 3 \
    --output projs \
    --plot-mode PYQTGRAPH
````
(Time on my machine: ~2 minutes, 55 seconds) 
The option `--plot-mode PYQTGRAPH` will display a plot window of the projection data after it is created. (This will not be available if the module was not installed with the `[gui]` option.)

Now, let's reconstruct with three different algorithms:
 1. Global Backprojection-Convolution (GBC)
 2. Filtered backprojection (FBP)
 3. Filtered backprojection with volumetric weightings (FBP-VW)

To perform a GBC reconstruction, we do:
```bash
aprict-cli recon gbc-cone-cyl \
    --geom projs/projgeom.json \
    --projs projs/projs.npy \
    --voxwid 0.5 \
    --volshape '(525,525,525)' \
    --alg GREWAR \
    --output recon_gbc \
    --plot-mode PYQTGRAPH
```
(Time on my machine: ~37 seconds)

To perform an FBP reconstruction, we do:
```bash
aprict-cli recon fbp-cone-cyl \
    --geom projs/projgeom.json \
    --projs projs/projs.npy \
    --voxwid 0.5 \
    --volshape '(525,525,525)' \
    --alg STANDARD \
    --output recon_fbp \
    --plot-mode PYQTGRAPH
```
(Time on my machine: ~12 seconds)

To perform an FBP-VM reconstruction, we change `--alg` to `NOOGREWAR`:
```bash
aprict-cli recon fbp-cone-cyl \
    --geom projs/projgeom.json \
    --projs projs/projs.npy \
    --voxwid 0.5 \
    --volshape '(525,525,525)' \
    --alg NOOGREWAR \
    --output recon_fbp-vw \
    --plot-mode PYQTGRAPH
```
(Time on my machine: ~22 seconds)


</details>


<details><summary><h2>Example B: fan-beam; circular source distribution</h2></summary>

Here's an example where we will use the CLI to simulate data from a FORBILD phantom resembling a double-chambered aluminium rock canister phantom.

Start by creating a clean directory to work in:
```bash
mkdir rockcantest
cd rockcantest
```

The first step is to create a `.json` file representing the fan-beam experiment geometry. This can be done like so:
```bash
aprict-cli make-geom fan-circ \
    --num 300 \
    --R 4 --L-minus-R 4 \
    --detwid 17.5 --nv 1 \
    --pixwid-isoc 0.002 --pixhei-isoc 0.002 \
    --output projgeom.json
```
Note the option `--nv 1`, which specifies that the detector height should be 1 pixel, which is appropriate for a fan-beam experiment, where we only intend to reconstruct a single z slice of the phantom. This is an alternative to supplying the `--dethei` option. Some options, such as these, are mutually exclusive. Use `aprict-cli make-geom fan-circ --help` to see detailed information about this.

Now, let's generate projection data from the phantom preset called `ROCK_CAN`. The `z=0` layer of the phantom is not all that interesting, so we'll also translate the input phantom down along the `z` axis by `-2.7` units (millimeters, in this case). Additionally, since we intend on simulating a 2D phantom here, we only want to sample rays at the vertical centre of each pixel. So, we will supply `--ssv 1` to ensure that only the middle of each pixel is used for ray end-points. The option `--ssu 100` ensures that 100 fanning rays are averaged to compute the projected attenuation at each pixel. This ensures greater accuracy, particularly in the presence of small features which may otherwise be missed by the pixel size.
```bash
aprict-cli project cbfp \
    --geom projgeom.json \
    --phantom-preset ROCK_CAN \
    --translate-vol '(0,0,-2.7)' \
    --ssu 100 \
    --ssv 1 \
    --output projs \
    --plot-mode PYQTGRAPH
````
(Time on my machine: ~1 minute, 46 seconds) 
The option `--plot-mode PYQTGRAPH` will display a plot window of the projection data (a sinogram) after it is created. (This will not be available if the module was not installed with the `[gui]` option.) 

You will find a copy of the `ROCK_CAN` phantom in the output directory `projs`, beside the projection data. The phantom is always copied to the output directory, to ensure a good record of the inputs to the simulation.

Finally, let's perform a reconstruction using a filtered backprojection (FBP) method for this geometry:
```bash
aprict-cli recon fbp-fan-circ \
    --geom projs/projgeom.json \
    --projs projs/projs.npy \
    --voxwid 0.002 \
    --volshape '(3000,3000,1)' \
    --alg STANDARD \
    --output recon_standard \
    --plotmin -0.25 \
    --plotmax 2.25 \
    --plot-mode PYQTGRAPH
```
(Time on my machine: ~2 seconds) 
Once again, the option `--plot-mode PYQTGRAPH` ensures that a plot window will pop up once the reconstruction is done. Note the `--plotmin -0.25` and `--plotmax 2.25` options here, which control the plotting range of the saved `*.png` files in the output directory. The whole volume is stored as a numpy array (`*.npy`), but it's useful to have some preview PNGs generated as well.


The previous reconstruction contains a lot of undersampling artefacts, due to the small number of projections (`300`). To a lesser extent, it also contains aliasing artefacts, due to the (de)magnified pixel size subceeding the voxel size. So, let's try an alternative algorithm that better handles this situation, at the expensive of reduced sharpness on the edges. This can be done by specifying `--alg NOOGREWAR`:
```bash
aprict-cli recon fbp-fan-circ \
    --geom projs/projgeom.json \
    --projs projs/projs.npy \
    --voxwid 0.002 \
    --volshape '(3000,3000,1)' \
    --alg NOOGREWAR \
    --output recon_noogrewar \
    --plotmin -0.25 \
    --plotmax 2.25 \
    --plot-mode PYQTGRAPH
```
(Time on my machine: ~2 seconds)

</details>


</details>
