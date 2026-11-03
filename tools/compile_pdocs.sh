#!/usr/bin/env bash
# Run this from the root of the git repo, with the python environment activated.

# I removed --math because that relies on mathjax, which fetches from another server. In the past, this has resulted in a security vulnerability. I prefer to keep the documents viewable offline, anyway.
rm -r docs 2> /dev/null
pdoc \
    -o docs/ \
    -t tools/pdoc_templates/ \
    --logo /img/logo_horz_big.svg \
    --favicon /img/logo_pic.svg \
    --footer-text "Copyright © 2026 Murdock Grewar. Licensed under GPLv3+." \
    ./src/aprict/ ./src/aprict/examples/ ./src/aprict/phantoms/FORBILD_phantoms.py
cp -r img docs/
