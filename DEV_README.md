Set up the dev environment in the following way.
 1. Install `uv` on your machine.
 2. Run these commands in the root directory of the git repository:
    uv sync --all-extras  # set up the python environment, including dev tools
    pre-commit install # install the pre-commit hooks, for automatic
                       # linting/formating/version incrementing, and license
                       # insertion.
To activate the Python environment, you can use `source .venv/bin/activate`. 
You can use `deactivate` to deactivate it.

The pre-commit hooks can be run on all staged files with `pre-commit run`.
The pre-commit hooks can be run on all staged files and all files already added
to the git repository with `pre-commit run --all-files`. Some hooks will care
about the files passed to them; others will not care (such as the version
incrementing).
