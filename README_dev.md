Set up the dev environment in the following way.
 1. Install `uv` on your machine.
 2. Run these commands in the root directory of the git repository:
    uv sync --all-extras  # set up the python environment, including dev tools
    uv run pre-commit install # install the pre-commit hooks, for automatic
                              # linting/formating/version incrementing, 
                              # license insertion, security testing,
                              # unit testing, et cetera.
To activate the Python environment, you can use `source .venv/bin/activate`. 
You can use `deactivate` to deactivate it.

The various pre-commit tests can be run without committing with `./verify.sh`.

The python environment should be activated when developing or building the module.

Important note on CUDA toolchain version: the versions of cuda-toolkit and 
cuda-bindings is controlled by the uv.lock file to ensure that the generated ptx
code is compatible with CUDA version 13.X. If the uv lock file is updated,
then the 13.X versions of these packages should be restored. E.g. the Python venv
should be updated like this:
```bash
uv lock -U -P cuda-toolkit~=13.0.0 -P cuda-bindings~=13.0.0
uv sync --all-extras
```
