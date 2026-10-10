This directory is used to store extra data.
In particular, the injected CUDA source code is generated and stored here.
CuPy loads these kernels.
The ./ptx and ./dtypes directories are caching directories, and are ignored
by git. They can be safely deleted, and the respective caches will regenerate.
