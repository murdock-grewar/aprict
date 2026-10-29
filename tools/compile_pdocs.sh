#!/usr/bin/env bash
# Run this from the root of the git repo, with the python environment activated.
pdoc -o docs/ -t tools/pdoc_templates/ ./src/aprict/ ./src/aprict/examples/ ./src/aprict/phantoms/FORBILD_phantoms.py ./src/aprict/volume/volume.py
