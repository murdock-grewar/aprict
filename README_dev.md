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
